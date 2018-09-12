import torch
import torch.nn as nn
from torch.utils.data import Dataset

import numpy as np
from skimage import io
import h5py as h5

import os

class LeMoNADe_VAE(nn.Module):
    def __init__(self, n_filter, filter_length,n_frames,n_pixel_x,n_pixel_y,lambda_1, dtype, device):
        super(LeMoNADe_VAE,self).__init__()
        
        self.n_filter = n_filter
        self.filter_length = filter_length
        self.n_frames = n_frames
        self.n_pixel_x = n_pixel_x
        self.n_pixel_y = n_pixel_y
        
        self.dtype = dtype
        self.device = device
        
        self.relu = nn.ReLU()
        self.elu = nn.ELU()
        self.softplus = nn.Softplus()
        
        # temperature for the relaxation of the variational posterior q(z|x)
        self.lambda_1 = lambda_1
        
        # layers of the encoder network
        # 2D convolutions operating on each frame separately
        self.conv1 = nn.Conv2d(1,24,(3,3))
        self.conv2 = nn.Conv2d(24,48,(3,3))
        self.conv3 = nn.Conv2d(48,72,(3,3))
        self.conv4 = nn.Conv2d(72,96,(3,3))
        self.conv5 = nn.Conv2d(96,120,(3,3))
        self.conv6 = nn.Conv2d(120,48,(1,1))
        
        # remaining image size after these convoltuions
        remaining_x = int((((self.n_pixel_x - 3 + 1 - 3 + 1)/2 - 3 + 1 - 3 + 1)/2 -3 + 1 - 1 + 1))
        remaining_y = int((((self.n_pixel_y - 3 + 1 - 3 + 1)/2 - 3 + 1 - 3 + 1)/2 -3 + 1 - 1 + 1))
        
        # final 3D convolution
        self.conv3d = nn.Conv3d(48, self.n_filter*2, (self.filter_length,remaining_x,remaining_y),padding=(int(self.filter_length-1),0,0))
        
        
        self.max_pool = nn.MaxPool2d((2,2),stride=2)#, return_indices=True)
        
        # single 3D deconvolution layer of the decoder network
        self.de_direct = nn.ConvTranspose3d(self.n_filter, 1, (self.filter_length,self.n_pixel_x,self.n_pixel_y),padding=(int(self.filter_length-1),0,0))
        
        
        
    def encoder(self,x):
        # define the encoder network 
        h1 = self.elu(self.conv1(x))
        h2 = self.elu(self.conv2(h1))
        h2m = self.max_pool(h2)
        h3 = self.elu(self.conv3(h2m))
        h4 = self.elu(self.conv4(h3))
        h4m = self.max_pool(h4)
        h5 = self.elu(self.conv5(h4m))
        h6 = self.elu(self.conv6(h5))
        out = self.softplus(self.conv3d(h6.transpose(0,1)[None, ...]))
        
        # split the n_filter*2 probability maps into the probabiltities for 
        # sampling a 1 (alpha_1) and sampling a 0 (alpha_2)
        alpha_1 = out[:,:self.n_filter]
        alpha_2 = out[:,self.n_filter:]
        return(alpha_1, alpha_2)
    
    def decoder(self,z):
        # define the decoder network
        x = self.relu(self.de_direct(z))[0].transpose(0,1)
        return(x)
    
    def repar(self,alpha_1,alpha_2):
        # reparametrization trick for the BinConcrete distributions
        T = alpha_1.data.size(2)
        M = alpha_1.data.size(1)
        # sample U from Uniform(0,1)
        eps = 1e-7
        U = torch.rand((M,T),dtype=self.dtype, device=self.device, requires_grad=False)
        
        self.alpha = alpha_1[0,:,:,0,0].div(alpha_2[0,:,:,0,0]+eps)
        y = (self.alpha.mul(U).div(1-U+eps)).pow(1/self.lambda_1)
        z = y.div(1+y).mul(alpha_1[0,:,:,0,0])
        return(z.view(alpha_1.size()))
    
    def forward(self,x):
        # forward pass
        alpha_1, alpha_2 = self.encoder(x)
        self.z = self.repar(alpha_1, alpha_2)
        x_ = self.decoder(self.z)
        return(x_)



#######################################
# loading the dataset
class VideoDataset(Dataset):
    """Load the dataset as a whole or process only a short sequence of it 
        in every iteration."""

    def __init__(self, mode, device, dtype, data_file, data_sheet, batch_length, filter_length):
        """
        Args:
            data_file (string): Path to the *.h5 or *.tif file with the video.
            data_sheet (string): Dataset within the *.h5 file.
            mode (string): Can either be 
                                        "complete" = process the video as a whole 
                                        or 
                                        "batches" = process only a short sequence of consecutive frames in every iteration. 
            batch_length (int): Number of consecutive frames processed in each epoch if mode is set to "batches"
            filter_length (int): Upper limit for the motif length
        """
        
        if os.path.exists(data_file+'.tif'):
            data = io.imread(data_file+'.tif').astype(np.float32)
            print('dataset ' + data_file+'.tif loaded.')
        elif os.path.exists(data_file+'.h5'):
            d = h5.File(data_file+'.h5','r')
            data = d[data_sheet][...].astype(np.float32)
            d.close()
            print('dataset ' + data_file+'.h5, sheet /'+data_sheet +' loaded.')
        else:
            raise Exception('DatasetNotFoundError: No dataset named '+ data_file + 
                            '.tif or ' + data_file + '.h5 found. ' +
                            'Please provide the dataset either as TIF stack or in a HDF5 file.')
        
        datastd = data.std()
        datamean = data.mean()
        data -= datamean
        data /= datastd
        
        if mode == 'complete':
            self.data = torch.tensor(data, dtype=dtype, device=device)
        else:
            self.data = torch.tensor(data, dtype=dtype, device='cpu')
            
        self.batch_length = batch_length
        self.filter_length = filter_length
        self.device = device
        self.mode = mode
        
        
        
    def __len__(self):
        return(1)
    
    def __pixels_x__(self):
        return(self.data.size(1))
    
    def __pixels_y__(self):
        return(self.data.size(2))
    
    def __real_len__(self):
        return(self.data.size(0))

    def __getitem__(self, idx):
        if self.mode == 'batches':
            if 10*self.filter_length < self.batch_length:
                start = np.random.choice(self.data.size(0)-10*self.filter_length,1)[0]
            else:
                start = np.random.choice(self.data.size(0)-self.batch_length,1)[0]
            
            if (start + self.batch_length) < len(self.data):
                sample = self.data[start:start+self.batch_length]
            else:
                sample = self.data[start:]
        else:
            sample = self.data
            
        return(sample.to(self.device))
        
    def __get_all__(self,idx):
        if self.mode == 'batches':
            start = idx * self.batch_length 
            if (start + self.batch_length) < len(self.data):
                sample = self.data[start:start+self.batch_length]
            else:
                sample = self.data[start:]
        else:
            sample = self.data
        return(sample[:,None].to(self.device))





        
        
        
