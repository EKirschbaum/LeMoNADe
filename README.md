# LeMoNADe: Learned Motif and Neuronal Assembly Detection

This is a toolbox to identify reoccurring neuronal firing patterns (aka motifs) in calcium imaging videos. 

## Publication

If you use this software in your research, please cite our publication:

**"LeMoNADe: Learned Motif and Neuronal Assembly Detection in calcium imaging videos"**, E. Kirschbaum, M. Haußmann, S. Wolf, H. Jakobi, J. Schneider, S. Elzoheiry, O.Kann, D. Durstewitz, F. A. Hamprecht, *arXiv preprint arXiv:1806.09963*, 2018. 
[[pdf]](https://arxiv.org/pdf/1806.09963.pdf)

## Requirements:

* [**Python 3.6 (or later)**](https://www.python.org/): we recommend installing it with [Anaconda](https://www.anaconda.com/download/)
* [**PyTorch 0.4 (or later)**](http://pytorch.org/): Make sure to install the Pytorch version for Python 3.6 with CUDA support. 
* **Additional Python packages**: numpy, matplotlib, h5py, os, argparse, skimage, PIL 

In Anaconda you can install with:
```
conda install numpy matplotlib h5py os argparse skimage PIL
```

## Usage

There are two recommended ways to use this code to analyze your data: 

### Analyze a dataset with one parameter configuration using the command line

1. Open a terminal window and navigate to the folder containing the file `lemonade.py`
2. Execute the file `lemonade.py` with the mandatory options.  
**Example:**
```
python lemonade.py -d path_to_dataset/data_file_name -M 3 -F 10 -a .1
```


### Analyze one or more datasets using a series of different parameter settings 

1. Open a terminal window and navigate to the folder containing the file `run_multiple_settings.py`.
2. Edit the file `run_multiple_settings.py` e.g. using   
```
pico run_multiple_settings.py
```   
to set the parameters you want to test. 
3. Execute the file using    
```
python run_multiple_settings.py
```

Instead of using the terminal you can of course also use an IDE like e.g. [Spyder](https://anaconda.org/anaconda/spyder) (included in Anaconda) to edit and run the files.


###Options   

####Mandatory Options  

| **Option** | **Name** | **Description** |  
|--------|-----|-----------|   
| `-d` | data file | Specify the complete path from the current folder to the location of the dataset. The dataset can be provided either as TIF stack or HDF5 file. Enter the (path and) name of the data file without the endig .tif or .h5. If you use a HDF5 file, make sure to also specify the correct sheet within the file by using the option `-ds` (see below).  |   
| `-M` | number of motifs | Specify the maximum number of motifs to look for. |    
| `-F` | motif length | Upper limit for the temporal extend of each motif. |   
| `-a` | sparsity parameter | Influences the sparsity of the found activations. Smaller values lead to more sparsity. |   

####Additional Options

| **Option** | **Name** | **Description** |  
|--------|-----|-----------|   
| `-ds` | data sheet | Only required if the data is provided in an HDF5 file. Specify the sheet within the file containing the CA video. |   
| `-kld`  | beta_kld | Additional sparsity parameter to regulate the influence of the KL-divergence term in the loss function. The influence of the KL-divergence on the sparsity of the found activation depends on the choice of `-a`. However, in most cases higher values for `-kld` will result in sparser activations. |   
| `-mode` | processing mode | The processing mode can be either set to "complete" or "batches". If it is set to "complete", the video is processed as a whole in every learning epoch. However, depending on the size of the video and the RAM of the GPU this is not always possible. In the mode "batches" only a shorter sequence of consecutive frames is analyzed in every epoch. |   
| `-b` | batch length | The number of consecutive frames processed in each epoch if the processing mode is set to "batches". This number should be as big as possible. If you get a memory error, it is too big. |    
| `-e` | epochs | Number of epochs used for training. More training epochs can improve the results but also enlarge computation time. Training can be terminated at any time using KeyboardInterrupt and the so far achieved results are saved. |   
| `-gpu` | GPU ID | If multiple GPUs are available you can specify the ID of the GPU you want to use. |   
| `-q` | quiet | Add `-q` without any further specification to turn off the plotting of the found motifs and activations. |
 
 
 
##Plotting 

To create additional plots of the results or *.tif and *.gif movies of the found motifs, you can use the functions provided in `plot.py`.      


  


