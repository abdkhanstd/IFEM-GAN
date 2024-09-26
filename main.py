import scripts.exdark_convert as ec

import argparse
import itertools

import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.autograd import Variable
from PIL import Image
import torch
from scripts.models import Generator
from scripts.models import Discriminator
from scripts.utils import ReplayBuffer
from scripts.utils import LambdaLR
from scripts.utils import Logger
from scripts.utils import weights_init_normal
from scripts.utils import loss_plot
from scripts.datasets import ImageDataset
import scripts.abdutils as abd
import numpy as np
from tqdm import tqdm

from PIL import ImageFile
from PIL import Image
import PIL
import os
from torch.optim.lr_scheduler import ReduceLROnPlateau


os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["ORCH_USE_CUDA_DSA"] = "1"

# Pillow needs to be tolerant for ExDark Dataset
ImageFile.LOAD_TRUNCATED_IMAGES = True

# Set the CUBLAS_WORKSPACE_CONFIG environment variable
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

import warnings

# Temporarily ignore all warnings
warnings.filterwarnings('ignore')

parser = argparse.ArgumentParser()
parser.add_argument('--epoch', type=int, default=0, help='starting epoch')
parser.add_argument('--dataset', type=str, default='RTTS_AUG', help='Choose dataset [ExDark, RTTS,ExDark_AUG, RTTS_AUG  CityScapesFoggy ]') # CityScapesFoggy: Best results on small. however, removed due to deadline
parser.add_argument('--n_epochs', type=int, default=2000, help='number of epochs of training')
parser.add_argument('--batchSize', type=int, default=20, help='size of the batches') # best on 32
parser.add_argument('--dataroot', type=str, default='datasets/lol', help='root directory of the dataset for Training GAN')
parser.add_argument('--lr', type=float, default=0.0002, help='initial learning rate')
parser.add_argument('--decay_epoch', type=int, default=10, help='epoch to start linearly decaying the learning rate to 0')
parser.add_argument('--size', type=int, default=512, help='size of the data crop (squared assumed)') # best on 128
parser.add_argument('--input_nc', type=int, default=3, help='number of channels of input data')
parser.add_argument('--output_nc', type=int, default=3, help='number of channels of output data')
parser.add_argument('--cuda', default=True, help='use GPU computation')
parser.add_argument('--n_cpu', type=int, default=8, help='number of cpu threads to use during batch generation')
parser.add_argument('--generator_S2O', type=str, default='weights/GAN/netG_A2B.pth', help='A2B generator checkpoint file')
parser.add_argument('--generator_O2S', type=str, default='weights/GAN/netG_B2A.pth', help='B2A generator checkpoint file')
parser.add_argument('--patience', type=int, default=30, help='number of channels of output data')

parser.add_argument('--YoloWeightsDir', type=str, default='weights/yolo8', help='Weights dir for YOLO8')


parser.add_argument('--trainGAN', default=False, help='Skip GAN training (default: False)')
parser.add_argument('--trainOD', default=False, help='Skip training for obejct detectore(default: False)')
parser.add_argument('--process', default=True, help='Organize the datasets first')
parser.add_argument('--GANInference', default=True, help='Organize the datasets first')
parser.add_argument('--GANWeights', default='weights/GAN', help='Load/Save folder for GAN Trainining and Inference weights')
parser.add_argument('--GANtemp', default='output/GAN_OPT', help='Temp folder to save inference')


opt = parser.parse_args()


abd.ClearScreen()
#abd.ShowUsage()
abd.SelectGPU()  # Select the GPU
abd.LookForKeys()  # Used to properly exit programs by freeing GPUs



# Override during debugging
opt.process=False
opt.GANInference=True
opt.trainGAN=False 
opt.trainOD=False
second_pass=True

opt.n_epochs=50
opt.decay_epoch=10
opt.size=512


if opt.size==256:
    opt.batchSize=4
elif opt.size==128:
    opt.batchSize=8
else:
    opt.batchSize=2


if opt.dataset=='RTTS':
    opt.size2=2000
else:
    opt.size2=2000

opt.cuda=True



abd.Delete('runs')


# Hyperparameters for learning rate reduction
lr_reduction_patience = 5  # epochs to wait before reducing lr if no improvement
lr_reduction_factor = 0.5  # factor to reduce lr

#
save_epoch_folder='output/GAN_epochs'
abd.Delete(save_epoch_folder)
abd.CreateFolder(save_epoch_folder,'c')

# A reminder to use GPU
if torch.cuda.is_available() and not opt.cuda:
    print("Info: You have a CUDA device, try running with --cuda")
##############################################################################################
#                           Training Our GAN                                                 #  
##############################################################################################
def color_loss(generated, target):
    """
    Compute the color loss between the generated image and the target image.

    Args:
    - generated (torch.Tensor): the generated image tensor.
    - target (torch.Tensor): the target image tensor.

    Returns:
    - loss (torch.Tensor): the color loss.
    """
    # Convert the generated and target images to grayscale
    generated_gray = torch.mean(generated, dim=1, keepdim=True)
    target_gray = torch.mean(target, dim=1, keepdim=True)

    # Compute the L2 (Euclidean) distance between the grayscale images
    loss = F.mse_loss(generated_gray, target_gray)
    return loss

def perceptual_loss(generated, target):
    """
    Compute the perceptual loss between the generated image and the target image.
    
    Args:
    - generated (torch.Tensor): the generated image tensor.
    - target (torch.Tensor): the target image tensor.

    Returns:
    - loss (torch.Tensor): the perceptual loss.
    """
    # Extract features from an intermediate layer for both images
    gen_features = vgg(generated)
    target_features = vgg(target)
    
    # Compute the MSE loss between the two sets of features
    loss = F.mse_loss(gen_features, target_features)
    return loss


if opt.trainGAN:
    print("Start training GAN ... ")


    from torchvision.models import vgg19
    import torch.nn.functional as F
    # Load a pre-trained VGG19 model and extract the desired feature layer for perceptual loss
    vgg = vgg19(pretrained=True).features.to('cuda:0').eval()
    
    
    # Freeze VGG parameters
    for parameter in vgg.parameters():
        parameter.requires_grad = False
    

    # Initialize Generators and Discriminators
    netG_A2B = Generator(opt.input_nc, opt.output_nc)
    netG_B2A = Generator(opt.output_nc, opt.input_nc)
    netD_A = Discriminator(opt.input_nc)
    netD_B = Discriminator(opt.output_nc)

    # Move models to GPU if available
    if opt.cuda:
        netG_A2B.cuda()
        netG_B2A.cuda()
        netD_A.cuda()
        netD_B.cuda()

    # Initialize model weights
    netG_A2B.apply(weights_init_normal)
    netG_B2A.apply(weights_init_normal)
    netD_A.apply(weights_init_normal)
    netD_B.apply(weights_init_normal)

    # Loss functions
    criterion_GAN = torch.nn.MSELoss()
    criterion_cycle = torch.nn.L1Loss()
    criterion_identity = torch.nn.L1Loss()

    # Optimizers & Learning Rate Schedulers
    optimizer_G = torch.optim.Adam(itertools.chain(netG_A2B.parameters(), netG_B2A.parameters()),
                                    lr=opt.lr, betas=(0.5, 0.999))
    optimizer_D_A = torch.optim.Adam(netD_A.parameters(), lr=opt.lr, betas=(0.5, 0.999))
    optimizer_D_B = torch.optim.Adam(netD_B.parameters(), lr=opt.lr, betas=(0.5, 0.999))

    lr_scheduler_G = torch.optim.lr_scheduler.LambdaLR(optimizer_G, lr_lambda=LambdaLR(opt.n_epochs, opt.epoch, opt.decay_epoch).step)
    lr_scheduler_D_A = torch.optim.lr_scheduler.LambdaLR(optimizer_D_A, lr_lambda=LambdaLR(opt.n_epochs, opt.epoch, opt.decay_epoch).step)
    lr_scheduler_D_B = torch.optim.lr_scheduler.LambdaLR(optimizer_D_B, lr_lambda=LambdaLR(opt.n_epochs, opt.epoch, opt.decay_epoch).step)

    # Input and target tensors
    Tensor = torch.cuda.FloatTensor if opt.cuda else torch.Tensor
    input_A = Tensor(opt.batchSize, opt.input_nc, opt.size, opt.size)
    input_B = Tensor(opt.batchSize, opt.output_nc, opt.size, opt.size)

    target_real = Variable(Tensor(opt.batchSize, 1, opt.size, opt.size).fill_(1.0), requires_grad=False)
    target_fake = Variable(Tensor(opt.batchSize, 1, opt.size, opt.size).fill_(0.0), requires_grad=False)


    # Replay buffers for fake images
    fake_A_buffer = ReplayBuffer()
    fake_B_buffer = ReplayBuffer()






    # Data transformations
    transforms_ = [
        transforms.Resize(int(opt.size * 1.15), Image.Resampling.BICUBIC),
        transforms.RandomResizedCrop(opt.size),  # Random cropping and resizing
        transforms.RandomHorizontalFlip(),  # Random horizontal flipping
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),  # Color jittering
        transforms.RandomRotation(15),  # Random rotation by up to 15 degrees
        transforms.RandomApply([transforms.GaussianBlur(5)], p=0.3),  # Rndom Gaussian blur
        transforms.ToTensor(),
        transforms.RandomAdjustSharpness(sharpness_factor=2, p=0.5),
        transforms.RandomErasing(p=0.2),
        transforms.RandomGrayscale(p=0.1),
        transforms.RandomAdjustSharpness(sharpness_factor=2, p=0.5),  
        
        # Rain and Fog augmentations are in dataloader


        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ]

    # Data loader    
    # unaligned=True for lol dataset.
    # Data loader for training
    dataloader = DataLoader(ImageDataset(opt.dataroot, transforms_=transforms_, unaligned=False, mode='train', is_validation=False), 
                            batch_size=opt.batchSize, shuffle=True, num_workers=opt.n_cpu)
    
    # Data loader for validation
    Valdataloader = DataLoader(ImageDataset(opt.dataroot, transforms_=transforms_, unaligned=False, mode='train', is_validation=True), 
                               batch_size=opt.batchSize, shuffle=True, num_workers=opt.n_cpu)


    # Loss and logger setup
    logger = Logger(opt.n_epochs, len(dataloader))
    lossPlot = loss_plot(opt.n_epochs)
    best_loss = float('inf')  
    best_epoch = 0
    early_stopping_patience = opt.patience
    early_stopping_counter=0
    
    
    
    # Initialize best losses for GAN A2B and GAN B2A
    best_loss_G_A2B = float('inf')
    best_loss_G_B2A = float('inf')

    best_loss_G=float('inf')

    # Trackers for early stopping
    stagnant_epochs_A2B = 0
    stagnant_epochs_B2A = 0

    # Create folder 
    
    abd.CreateFolder(opt.GANWeights,'c')

    scheduler_G = ReduceLROnPlateau(optimizer_G, mode='min', factor=0.1, patience=5, verbose=True)
    real_A = Tensor(opt.batchSize, opt.input_nc, opt.size, opt.size)
    real_B = Tensor(opt.batchSize, opt.output_nc, opt.size, opt.size)    


    best_loss = float('inf')  
    early_stopping_counter = 0
    lambda_perceptual = 0.2

    # Training Loop
    for epoch in range(opt.epoch, opt.n_epochs):
        epoch_losses_G_A2B = []
        epoch_losses_G_B2A = []  
            
        total_cycle_loss = 0
        total_generator_loss = 0              
        for i, batch in enumerate(dataloader):
            
            
            if batch is None:
                continue  # Skip None batches to avoid errors
            # Set model input
            #print(f"Batch {i + 1}: A shape = {batch['A'].shape}, B shape = {batch['B'].shape}")
    
            current_batch_size = batch['A'].shape[0]

            # Set model input with slicing to handle variable batch sizes
            real_A[:current_batch_size] = batch['A']
            real_B[:current_batch_size] = batch['B']

            #real_A = Variable(input_A.copy_(batch['A']))
            #real_B = Variable(input_B.copy_(batch['B']))

            optimizer_G.zero_grad()

            # G_A2B(B) should equal B if real B is fed
            same_B = netG_A2B(real_B)
            loss_identity_B = criterion_identity(same_B, real_B)*5.0
            # G_B2A(A) should equal A if real A is fed
            same_A = netG_B2A(real_A)
            loss_identity_A = criterion_identity(same_A, real_A)*5.0

            # GAN loss
            fake_B = netG_A2B(real_A)
            pred_fake = netD_B(fake_B)
            loss_perceptual_B = perceptual_loss(fake_B, real_B) * lambda_perceptual

            
            # Ensure that the target_real has the same shape as pred_fake
            target_real = Variable(Tensor(opt.batchSize).fill_(1.0), requires_grad=False)
            target_fake = Variable(Tensor(opt.batchSize).fill_(0.0), requires_grad=False)

            loss_GAN_A2B = criterion_GAN(pred_fake, target_real)

            fake_A = netG_B2A(real_B)
            pred_fake = netD_A(fake_A)

            loss_GAN_B2A = criterion_GAN(pred_fake, target_real)

            # Cycle loss
            recovered_A = netG_B2A(fake_B)
            loss_cycle_ABA = criterion_cycle(recovered_A, real_A)*10.0

            recovered_B = netG_A2B(fake_A)
            loss_cycle_BAB = criterion_cycle(recovered_B, real_B)*10.0


            # Total loss
            loss_perceptual_A = perceptual_loss(fake_A, real_A) * lambda_perceptual

            loss_color = color_loss(fake_B, real_B)
            # Add the color loss to the total loss
            loss_G = (
                loss_identity_A + loss_identity_B + 
                loss_GAN_A2B + loss_GAN_B2A +
                loss_cycle_ABA + loss_cycle_BAB + loss_perceptual_A + loss_perceptual_B + loss_color
            )


            loss_G.backward()
            
            optimizer_G.step()

            #Discriminator A 
            optimizer_D_A.zero_grad()

            # Real loss
            pred_real = netD_A(real_A)
            loss_D_real = criterion_GAN(pred_real, target_real)

            # Fake loss
            fake_A = fake_A_buffer.push_and_pop(fake_A)
            pred_fake = netD_A(fake_A.detach())
            loss_perceptual_A = perceptual_loss(fake_A, real_A) * lambda_perceptual
            
            loss_D_fake = criterion_GAN(pred_fake, target_fake)

            # Total discriminator loss for A
            loss_D_A = (loss_D_real + loss_D_fake)*0.5
            loss_D_A.backward()

            optimizer_D_A.step()

            # Discriminator B 
            optimizer_D_B.zero_grad()

            # Real loss
            pred_real = netD_B(real_B)
            loss_D_real = criterion_GAN(pred_real, target_real)
            
            # Fake loss
            fake_B = fake_B_buffer.push_and_pop(fake_B)
            pred_fake = netD_B(fake_B.detach())
            loss_D_fake = criterion_GAN(pred_fake, target_fake)

            # Total discriminator loss for B
            loss_D_B = (loss_D_real + loss_D_fake)*0.5
            loss_D_B.backward()

            optimizer_D_B.step()
            
            


            # Progress report (http://localhost:8097)
            '''
            logger.log({'🎈 loss_G': loss_G, 'loss_G_identity': (loss_identity_A + loss_identity_B), 'loss_G_GAN': (loss_GAN_A2B + loss_GAN_B2A),
                        'loss_G_cycle': (loss_cycle_ABA + loss_cycle_BAB), 'loss_D': (loss_D_A + loss_D_B)}, 
                        images={'real_A': real_A, 'real_B': real_B, 'fake_A': fake_A, 'fake_B': fake_B})
                        
            '''

            logger.log({'🎈 loss_G': loss_G, 'loss_G_GAN': (loss_GAN_A2B + loss_GAN_B2A),
            'loss_G_cycle': (loss_cycle_ABA + loss_cycle_BAB), 'loss_D': (loss_D_A + loss_D_B)}, 
            images={'real_A': real_A, 'real_B': real_B, 'fake_A': fake_A, 'fake_B': fake_B})

            lossPlot.accumulate_loss(epoch, loss_G, loss_D_A, loss_D_B)
            
            
            # Total discriminator loss for A
            loss_D = (loss_D_real + loss_D_fake)*0.5
            total_loss = (loss_G + loss_D) / 2
            
            
            # Collect losses for each epoch to average later
            epoch_losses_G_A2B.append(loss_GAN_A2B.item())
            epoch_losses_G_B2A.append(loss_GAN_B2A.item())
            
            total_cycle_loss += (loss_cycle_ABA.item() + loss_cycle_BAB.item())
            total_generator_loss += loss_G.item()            

        print("⌛ Evaluating model...")
        # Switch models to evaluation mode for validation
        netG_A2B.eval()
        netG_B2A.eval()
        netD_A.eval()
        netD_B.eval()

        import tqdm

        total_val_loss = 0.0
        with torch.no_grad():
            # Initialize the tqdm progress bar
            pbar = tqdm.tqdm(enumerate(Valdataloader), total=len(Valdataloader), desc="Validating")
            
            for i, batch in pbar:
                if batch is None:
                    continue

                current_batch_size = batch['A'].shape[0]
                real_A[:current_batch_size] = batch['A']
                real_B[:current_batch_size] = batch['B']

                # Forward pass through generators
                fake_B = netG_A2B(real_A)
                fake_A = netG_B2A(real_B)
                recovered_A = netG_B2A(fake_B)
                recovered_B = netG_A2B(fake_A)

                # Validation losses
                loss_identity_B = criterion_identity(netG_A2B(real_B), real_B) * 5.0
                loss_identity_A = criterion_identity(netG_B2A(real_A), real_A) * 5.0
                loss_GAN_A2B = criterion_GAN(netD_B(fake_B), target_real)
                loss_GAN_B2A = criterion_GAN(netD_A(fake_A), target_real)
                loss_cycle_ABA = criterion_cycle(recovered_A, real_A) * 10.0
                loss_cycle_BAB = criterion_cycle(recovered_B, real_B) * 10.0
                loss_perceptual_A = perceptual_loss(fake_A, real_A) * lambda_perceptual
                loss_perceptual_B = perceptual_loss(fake_B, real_B) * lambda_perceptual
                loss_color = color_loss(fake_B, real_B)

                val_loss_G = (
                    loss_identity_A + loss_identity_B +
                    loss_GAN_A2B + loss_GAN_B2A +
                    loss_cycle_ABA + loss_cycle_BAB +
                    loss_perceptual_A + loss_perceptual_B + loss_color
                )

                total_val_loss += val_loss_G.item()
                
                # Update the progress bar with the current loss value
                pbar.set_postfix({"📉 Validation Loss": total_val_loss / (i + 1)})

        # Calculate and print average validation loss
        avg_val_loss = total_val_loss / len(Valdataloader)
        print(f'Epoch {epoch+1}: 📉Avg. Validation Loss : {avg_val_loss}')

        # Early stopping and model checkpointing based on validation loss
        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            stagnant_epochs = 0
            # Save model checkpoints
            torch.save(netG_A2B.state_dict(), opt.GANWeights+'/netG_A2B.pth')
            torch.save(netG_B2A.state_dict(), opt.GANWeights+'/netG_B2A.pth')
            torch.save(netD_A.state_dict(), opt.GANWeights+'/netD_A.pth')
            torch.save(netD_B.state_dict(), opt.GANWeights+'/netD_B.pth')
            print("✅ Models Saved...")
        else:
            stagnant_epochs += 1
            print("❎ No improvement, so not saving Saved...")
            if stagnant_epochs >= early_stopping_patience:
                print("🛠️ Early stopping triggered...")
                break

        # Switch models back to training mode
        netG_A2B.train()
        netG_B2A.train()
        netD_A.train()
        netD_B.train()

    torch.cuda.empty_cache() 
        
        
##############################################################################################
#                                    Convert and Stabalize dataset                           #
##############################################################################################


if opt.dataset=='ExDark':
    # Process and prepare ExDark Dataset for Yolo8        
    if opt.process:
        ec.ProcessExDarkForYolo()
    

##############################################################################################
#                                    Inference Our GAN                                       #
##############################################################################################
import pywt

def wavelet_denoising(channel):
    coeffs = pywt.wavedec2(channel, 'db1', level=1)
    coeffs_H = list(coeffs)
    coeffs_H[0] *=1  # Set the approximation coefficients to zero.
    # Reconstruct the channel using the modified coefficients.
    channel_H = pywt.waverec2(coeffs_H, 'db1')
    return channel_H

if opt.GANInference:
    if opt.dataset=='ExDark' or opt.dataset=='RTTS' or opt.dataset=='RTTS_AUG' or opt.dataset=='ExDark_AUG':
        # Process and prepare ExDark Dataset for Yolo8
        print("Loading models and processing images..")
        
            
        # Inference with the trained GAN model
        # Networks
        netG_A2B = Generator(opt.input_nc, opt.output_nc)
        #netG_B2A = Generator(opt.output_nc, opt.input_nc)

        if opt.cuda:
            netG_A2B.cuda()

        # Load state dicts
        
        print("Loading points from : ",opt.generator_S2O)
        netG_A2B.load_state_dict(torch.load(opt.generator_S2O))

        # Set model's test mode
        netG_A2B.eval()

        abd.Delete(opt.GANtemp)
    
        # Overriding these value from experience gained from expriments
        opt.batchSize=1 #Must keep batch size 1 for inference
        opt.size=opt.size2
        
        # making a temp copy for inference
        abd.Delete(f'datasets/{opt.dataset}2')
        abd.Copy(f'datasets/{opt.dataset}', f'datasets/{opt.dataset}2')
        abd.Delete(f'datasets/{opt.dataset}2/images/test')           
        abd.Delete(f'datasets/{opt.dataset}2/images/train')
        
        #Also these 
        abd.Delete('datasets/lol/testA')
        abd.Delete('datasets/lol/testB')           
        

        for type in ['test', 'train']:   
            
            output_folder=opt.GANtemp+'/'+type
            abd.CreateFolder(output_folder,'c')
            
            # We need to copy the files to respective folders
            abd.Copy(f'datasets/{opt.dataset}/images/{type}', 'datasets/lol/testA')
            abd.Copy(f'datasets/{opt.dataset}/images/{type}', 'datasets/lol/testB')
     
            # Inputs & targets memory allocation
            Tensor = torch.cuda.FloatTensor if opt.cuda else torch.Tensor
            input_A = Tensor(opt.batchSize, opt.input_nc, opt.size, opt.size)
            input_B = Tensor(opt.batchSize, opt.output_nc, opt.size, opt.size)

            # Dataset loader
            transforms_ = [ transforms.ToTensor(),
                            transforms.Resize((opt.size, opt.size), Image.BICUBIC),               
                            transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5)) 
                            ]

            dataloader = DataLoader(ImageDataset(opt.dataroot, transforms_=transforms_, mode='test', Inference=True), 
                                    batch_size=opt.batchSize, shuffle=False, num_workers=opt.n_cpu)




            pbar = tqdm(enumerate(dataloader), total=len(dataloader))

            for i, batch in pbar:
                
                # Set model input
                real_A = Variable(input_A.copy_(batch['A']))
                #real_B = Variable(input_B.copy_(batch['B']))

                # Get the original image sizes from the batch data
                original_size_A = batch['original_size_A']
                original_size_B = batch['original_size_B']
                
                FileName = batch['FileName']
                FullFileName = batch['FullFileName']
                pbar.set_description(f"🚀 Processing With GAN ({type} images) : {FullFileName[0]}")

                # Generate output
                fake_B = 0.5 * (netG_A2B(real_A).data + 1.0)
                if second_pass:
                    #fake_B = 0.5 * (netG_A2B(fake_B).data + 1.0)
                    x4334=1

                
                #fake_A = 0.5*(netG_B2A(real_B).data + 1.0)
                
                # Convert the generated images to PIL Images
                fake_B = fake_B.squeeze().cpu()
                #fake_A = fake_A.squeeze().cpu()

         

                
                fake_B = transforms.ToPILImage()(fake_B)
                #fake_A = transforms.ToPILImage()(fake_A)
                
                
                #fake_A = fake_A.resize(original_size_A, Image.LANCZOS)
                fake_B = fake_B.resize(original_size_A, Image.LANCZOS)

                
                # Convert PIL Images to numpy arrays for element-wise operations
                #np_fake_A = np.array(fake_A)
                np_fake_B = np.array(fake_B)
                
                # Ensure the images are in float type for averaging
                #np_fake_A = np_fake_A.astype(np.float32)
                np_fake_B = np_fake_B.astype(np.float32)
                
                np_fake_A = np_fake_B
                                
                #np_fake_A = abd.copy_brighter_pixels(np_fake_A, np_fake_B)
                np_fake_A_cpy=np_fake_A
                #np_fake_A = abd.copy_brighter_pixels_percentage(np_fake_A, np_fake_B,70) 


                import cv2
                # Apply bilateral filter
                diameter = 100        # Diameter of each pixel neighborhood
                sigma_color = 75     # Filter sigma in the color space
                sigma_space = 75     # Filter sigma in the coordinate space

                #np_fake_A = cv2.bilateralFilter(np_fake_A, diameter, sigma_color, sigma_space)                
                channels = cv2.split(np_fake_A)

                np_fake_A = Image.fromarray(np.uint8(np_fake_A))


                
                #np_fake_A=abd.GaussianBlurImage(np_fake_A,1.1,verbose=False)  
                #np_fake_A=abd.SharpenImage(np_fake_A,40,verbose=False)

                # Process each channel with wavelet denoising.
                denoised_channels = []
                for channel in channels:
                    denoised = wavelet_denoising(channel)
                    denoised_channels.append(denoised)

                # Merge the denoised channels back together.
                denoised_image = cv2.merge(denoised_channels)
                
                                
                # Convert the floating-point image to 8-bit unsigned integer.
                np_fake_A = np.clip(denoised_image, 0, 255).astype('uint8')  
                
                
                              
                np_fake_A = Image.fromarray(np.uint8(np_fake_A))
                
                
                

                file_name= output_folder+'/'+ FileName[0]
                abd.SaveImage(np_fake_A, file_name)

                #Copying to Exdark dataset folder

            abd.Copy(output_folder, f'datasets/{opt.dataset}2/images/{type}')
                
            # We need to delete old folders and then copy the files to respective folders
            abd.Delete('datasets/lol/testA')
            abd.Delete('datasets/lol/testB')
            

##############################################################################################
#                                    YOLO8                                                   #
##############################################################################################
from ultralytics import YOLO
if opt.trainOD:
    
    print("Training Yolo on our GAN processed images...")

    # Delete old results first
    abd.Delete("yolo")
    abd.Delete("runs")

    
    import warnings

    # Ignore all warnings
    warnings.simplefilter("ignore")
    YoloWeightsPath=opt.YoloWeightsDir
    YoloPretrained='coco'  # Choices coco oi7
    YoloModel='yolov8x' # Choices yolov8l, yolov8m, yolov8n, yolov8s, yolov8x

    model = YOLO(f'{YoloWeightsPath}/{YoloPretrained}/{YoloModel}.pt')  # load a pretrained model


    results = model.train(data=f'{opt.dataset}.yaml', epochs=5000, imgsz=800, augment=True)#,batch=16,optimizer='Adam',patience=100, lr0=0.0002, momentum=0.9, weight_decay= 0.0005)

    abd.CreateFolder(f'weights/OD/{opt.dataset}','c')
    abd.Copy(f'runs/detect/train/weights/best.pt',f'weights/OD/{opt.dataset}')
    
    
else:
    abd.Delete('/home/abdkhan/BW3/datasets/RTTS2/labels/test.cache')
    
    model = YOLO(f'weights/OD/{opt.dataset}/best.pt')  # load a pretrained model
    metrics = model.val(data=f'{opt.dataset}.yaml', plots=True)# save_txt=True,classes=['Bicycle', 'Bus', 'Car', 'Motorcycle', 'Person'])  # evaluate model performance on the validation set


    # names={0: 'Bicycle', 1: 'Bus', 2: 'Car', 3: 'Motorcycle', 4: 'Person'}  # Needed for RTTS (utils/plotting.py and metrics.py)
    

'''

Ultralytics YOLOv8.0.20 🚀 Python-3.8.18 torch-1.8.0+cu111 CUDA:0 (NVIDIA GeForce RTX 3090, 24260MiB)
yolo/engine/trainer: task=detect, mode=train, model=yolov8x.yaml, data=data.yaml, epochs=5000, patience=50, 
batch=8, imgsz=1300, save=True, cache=False, device=, workers=8, project=None, name=None, exist_ok=False, 
pretrained=False, optimizer=SGD, verbose=False, seed=0, deterministic=True, single_cls=False, image_weights=False, rect=False, 
cos_lr=False, close_mosaic=10, resume=False, overlap_mask=True, mask_ratio=4, dropout=False, val=True, save_json=False, 
save_hybrid=False, conf=0.001, iou=0.7, max_det=300, half=False, dnn=False, plots=False, source=ultralytics/assets/, 
show=False, save_txt=False, save_conf=False, save_crop=False, hide_labels=False, hide_conf=False, vid_stride=1, line_thickness=3, 
visualize=False, augment=False, agnostic_nms=False, classes=None, retina_masks=False, boxes=True, format=torchscript, keras=False, optimize=False, 
int8=False, dynamic=False, simplify=False, opset=17, workspace=4, nms=False, lr0=0.01, lrf=0.01, momentum=0.937, weight_decay=0.0010078125, 
warmup_epochs=3.0, warmup_momentum=0.8, warmup_bias_lr=0.1, box=7.5, cls=0.5, dfl=1.5, fl_gamma=0.0, label_smoothing=0.0, nbs=64, hsv_h=0.015, hsv_s=0.7, hsv_v=0.4, 
degrees=0.0, translate=0.1, scale=0.9, shear=0.0, perspective=0.0, flipud=0.0, fliplr=0.5, mosaic=1.0, mixup=0.15, copy_paste=0.3, cfg=None, v5loader=False, save_dir=runs/detect/train
Overriding model.yaml nc=80 with nc=12


'''




