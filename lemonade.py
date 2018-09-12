
# LeMoNADe: Learned Motiv and Neuronal Assembly Detection in calcium imaging videos
# VAE framework to extract repeating patterns from calcium imaging videos

import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader

import numpy as np
import h5py as h5

import os

import argparse


from model import LeMoNADe_VAE, VideoDataset
import plot 

#########################

class LeMoNADe(object):
    def __init__(self, data_file, n_filter, filter_length, a,
                 data_sheet='ca_video', beta_kld=.1, 
                 mode='toobig', batch_length=500, 
                 epochs=10000, gpu=None, quiet=False):
        
        
        self.dtype = torch.float
        if gpu is not None:
            self.device = torch.device("cuda:"+str(gpu) if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.data_file = data_file
        self.data_sheet = data_sheet
        
        self.mode = mode
        if self.mode not in ['complete','batches']:
            raise Exception('ModeError: No valid mode given. Valid modes are: "complete" and "batches".')
            
        self.batch_length = batch_length
         
        self.n_filter = n_filter
        self.filter_length = filter_length
        
        self.a = a
        
        self.lambda_1 = .6
        self.lambda_2 = .5
        
        self.learning_rate = 1e-5
        
        self.beta_kld = beta_kld
        self.epochs = epochs
        
        # get the dataset
        print('loading the dataset...')
        self.dataset = VideoDataset(self.mode,self.device,self.dtype,self.data_file,
                                    self.data_sheet, self.batch_length, self.filter_length)
        
        self.n_frames = self.dataset.__real_len__()
        self.n_pixel_x = self.dataset.__pixels_x__()
        self.n_pixel_y = self.dataset.__pixels_y__()
        
        print('video dimensions: ' + str(self.n_frames) + ' frames, ' + str(self.n_pixel_y) + 'x' + str(self.n_pixel_x) + ' pixels')
                            
        
        self.folder, self.ending = self.build_names()
        
        self.quiet = quiet 
        
        self.lemonade = LeMoNADe_VAE(n_filter,filter_length,self.n_frames,
                                     self.n_pixel_x, self.n_pixel_y,
                                     self.lambda_1, self.dtype, self.device).to(self.device)
        
        
    def build_names(self):
        # create folder for outputs, if not already exits
        folder = './' + self.data_file + '/'
        if not os.path.exists(folder):
            os.makedirs(folder)
            
        # create endings of output file names
        ending = '_M' + str(self.n_filter) + '_F' + str(self.filter_length) + '_a' + str(self.a)    
        ending += '_kld' + str(self.beta_kld) + '_lam1' + str(self.lambda_1) + '_lam2' + str(self.lambda_2)
        if self.mode == 'batches' or self.mode == 'toobig':
            ending += '_b' + str(self.batch_length)
        ending += '_ep' + str(self.epochs) + '_lr' + str(self.learning_rate)
        
        if os.path.exists(folder + '/results' + ending + '.h5'):
            i = 1
            while os.path.exists(folder + '/results' + ending + '_' + str(i) + '.h5'):
                i += 1
            ending += '_' + str(i)
                
        return(folder, ending)
    
    def KLD_function(self, alpha):
            
        # KL regularization
        eps = 1e-7
        S = 1000
        N = alpha.size(0) * alpha.size(1) 
        # sampled from Uniform[0,1)
        U = torch.rand((alpha.size(0),alpha.size(1),S),dtype=self.dtype,device=self.device,requires_grad=False) 
        alpha_new = alpha.view(alpha.size(0),alpha.size(1),1)
        alpha_expanded = alpha_new.expand(alpha.size(0),alpha.size(1),S)
        one = torch.ones(alpha.size(),dtype=self.dtype, device=self.device, requires_grad=False)
            
        # log(a*lambda_2 / lambda_1) + 2
        fixed = torch.log(one.mul(self.a * self.lambda_2 / self.lambda_1)).add_(2.) 
            
        # -lambda_2 / lambda_1 * log(alpha)
        log_alpha = torch.log(alpha+eps).mul(-self.lambda_2 / self.lambda_1)
            
        # -2/S sum_s log(a * (alpha * U / (1-U))^(-lambda_2 / lambda_1) + 1)
        integral = torch.sum(torch.log((((U).div(1-U+eps)).mul(alpha_expanded)).add(eps).pow(-self.lambda_2/self.lambda_1).mul(self.a).add(1.)),dim=2).div(S).mul(-2.)            
            
            
        KLD = torch.sum(fixed.mul_(-1).add_(log_alpha.mul_(-1)).add_(integral.mul_(-1))).div_(N)
            
        return(KLD)
    
    def train(self):
        # training loop:
        for epoch in range(self.epochs):
                
            self.optimizer.zero_grad()   # zero the gradient buffers
                
            dataloader = DataLoader(self.dataset, batch_size=1,
                        shuffle=False, num_workers=0,drop_last=False)
                
            for i_batch, sample in enumerate(dataloader):
                input_ = sample.transpose(0,1)
            
            output = self.lemonade(input_)
                
            loss = self.recon_function(output,input_) + self.beta_kld * self.KLD_function(self.lemonade.alpha) 
                
            loss.backward()
            del output, input_, self.lemonade.alpha
                
                                
            print(epoch,"{:10.7f}".format(loss.item()))
                
            self.optimizer.step()    # Does the update
    
        return()
    
    def get_results(self):
        
        with torch.no_grad():
            if self.mode == 'batches' or self.mode == 'toobig':
                all_z = []
                
                if self.dataset.__real_len__()/self.batch_length > int(self.dataset.__real_len__()/self.batch_length):
                    steps = int(self.dataset.__real_len__()/self.batch_length) + 1
                else:
                    steps = int(self.dataset.__real_len__()/self.batch_length)
                for idx in range(steps):
                    self.lemonade.forward(self.dataset.__get_all__(idx)).data.cpu().numpy()[:,0]
                    z = self.lemonade.z.data.cpu().numpy()[:,:,self.filter_length-1:]
                    all_z.append(z)
                    del z
                
                z = np.concatenate(all_z, axis=2)            
            else:
                self.lemonade.forward(self.dataset.__get_all__()).data.cpu().numpy()[:,0]
                z = self.lemonade.z.data.cpu().numpy()
                
            motifs = self.lemonade.relu(self.lemonade.de_direct.weight)[:,0].data.cpu().numpy()
     
        return(motifs,z)
        
    def run(self):  
  
        # create your optimizer
        self.optimizer = optim.Adam(self.lemonade.parameters(), lr=self.learning_rate)
        
        # define loss function
        self.recon_function = nn.MSELoss()
        
        try:
            print('start training the VAE...')
            
            self.train()    
                
        except KeyboardInterrupt: 
            pass
        except: 
            raise 
        
        
        print('training terminated.\nsaving the results...')
        
        motifs, z = self.get_results()
        
        with h5.File(self.folder+'results'+self.ending+'.h5','a') as o:
            o.create_dataset('final_motifs',data=motifs)
            o.create_dataset('final_z',data=z[0,:,:,0,0])
        
        
        if not self.quiet:
            print('plotting...')
            plotter = plot.make_plots(self.folder, self.ending)
            plotter.plot_motifs(motifs)
            plotter.plot_z(z[0,:,:,0,0])
            #plotter.make_motif_movies(motifs)      # creates a GIF and a TIF stack for each motif
            
        print('All done.')
        return()










if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='learned motif and neuronal assembly detection')
    
    parser.add_argument('-d', '--data_file', default="data_file", help="file with the CA video as a *.tif stack or *.h5 file, enter file name without ending .tif or .h5 (default: %(default)s)")
    parser.add_argument('-ds', '--data_sheet', default="ca_video", help="sheet within the *.h5 file containing the CA video, not necessary if data is saved as TIF stack (default: %(default)s)")
    
    parser.add_argument('-M', '--motifs', type=int, default=3, help="max number of motifs to be found (default: %(default)d)")
    parser.add_argument('-F', '--motif_length', type=int, default=10, help="max temporal length of the motif (default: %(default)d")
    parser.add_argument('-a', '--a', type=float, default=.1, help="a (default: %(default)d)")
    
    parser.add_argument('-kld', '--beta_kld', type=float, default=0.1, help="beta_kld (default: %(default)d)")
    
    parser.add_argument('-mode', '--mode', default="batches", help="mode in which the data is processed, possible options are: 'batches' and 'complete' (default: %(default)s).")
    parser.add_argument('-b', '--batch_length', type=int, default=500, help="number of frames to be processed in each epoch (default: %(default)d)")
    
    parser.add_argument('-e', '--epochs', type=int, default=10000, help="number of training epochs (default: %(default)d)")    
    
    parser.add_argument('-gpu','--gpu', type=int, default=None, help="ID of the GPU to be used (default: %(default)d)")
    
    parser.add_argument('-q', '--quiet', action='store_true', help="Add -q for NO pictures.")
 
    
    
        
    args = parser.parse_args()
    

    
    data_file = args.data_file
    data_sheet = args.data_sheet
    
    M = args.motifs 
    
    F = args.motif_length 
    if F % 2 == 0:
        F += 1
    
    a = args.a
    
    beta_kld = args.beta_kld
    
    mode = args.mode
    
    batch_length = args.batch_length
    
    epochs = args.epochs
    
    gpu_id = args.gpu
    
    quiet = args.quiet 
    
    
    
    lemonade = LeMoNADe(data_file=data_file, n_filter=M, filter_length=F, a=a, 
                        data_sheet=data_sheet, beta_kld=beta_kld, 
                        mode=mode, batch_length=batch_length, 
                        epochs=epochs, gpu=gpu_id, quiet=quiet)
    
    lemonade.run()
         
    
    
    
    