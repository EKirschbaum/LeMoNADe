# Run a series of multiple experiments with different parameter settings 

from lemonade import LeMoNADe


data_files = ['data_file']     # list of data files to be analyzed 
data_sheet = ''                # sheet within the *.h5 file containing the video, irrelevant if data file is a *.tif
    
M_ = [3,5,10]   # list of the numbers of motifs to look for             
    
F_ = [10,20]    # list of the motif lenghts to test
    
a_ = [0.1,0.05] # list of the sparsity parameters to test
        
mode = 'batches'    # processing mode
    
batch_length = 500  # number of consecutive frames processed in each epoch
    
epochs = 10000      # number of training epochs
        
gpu_id = 0          # ID of the GPU to be used
        
quiet = False       # set to True to omit plots of the found motifs and activations 


# For all datasets listed above, test all parameters listed above. 
for data_file in data_files:    
    for M in M_:
        for F in F_:
            if F % 2 == 0:
                F += 1
            for a in a_:
    
        
                lemonade = LeMoNADe(data_file=data_file, n_filter=M, filter_length=F, a=a,
                            data_sheet=data_sheet, mode=mode, batch_length=batch_length, 
                            epochs=epochs, gpu=gpu_id, 
                            quiet=quiet)
        
                lemonade.run()
             
        
        
        
        