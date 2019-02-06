
import numpy as np
import matplotlib
matplotlib.use("agg")
import matplotlib.pyplot as plt
plt.switch_backend('agg')

from PIL import Image



    
class make_plots(object):
    def __init__(self, folder, ending, eps=False):
        self.folder = folder
        self.ending = ending
        self.eps = eps
        
    
    def plot_loss(self,loss):
        params = {
                'axes.labelsize': 50,
                'font.size': 50,
                'legend.fontsize': 30,
                'xtick.labelsize': 20,
                'ytick.labelsize': 20,
                "text.usetex": False,
                'figure.figsize': [10, 10]
                }
            
        plt.rcParams.update(params)
        fig, ax = plt.subplots()
        ax.set_ylabel('loss')
        ax.set_xlabel('epoch')
        ax.plot(np.arange(len(loss)),loss,'r-')
        fig.savefig(self.folder+'loss'+self.ending+'.png')
        return()
    
    
    def plot_motifs(self, x):
        n_filter = x.shape[0]
        filter_length = x.shape[1]
        if filter_length < 50:        
            width = filter_length*7.
            height = n_filter*7.
        else:
            width = filter_length*5.
            height = n_filter*5.
            
        params = {
                'axes.labelsize': 50,
                'font.size': 50,
                'legend.fontsize': 30,
                'xtick.labelsize': 20,
                'ytick.labelsize': 20,
                "text.usetex": False,
                'figure.figsize': [width, height]
                }
            
        plt.rcParams.update(params)
                       
        fig, ax = plt.subplots(n_filter,filter_length)
        if n_filter == 1:
            if filter_length == 1:
                ax.set_ylabel('motif 0')
                ax.set_xlabel('frame 0')
                ax.set_yticks([])
                ax.set_yticklabels([])
                ax.set_xticks([])
                ax.set_xticklabels([])
                ax.imshow(x[0,0], vmin=np.min(x[0]), vmax=np.max(x[0]), cmap=plt.cm.binary)
            else:
                ax[0].set_ylabel('motif 0')
                for i in range(filter_length):
                    ax[i].set_xlabel('frame '+str(i))
                    ax[i].set_yticks([])
                    ax[i].set_yticklabels([])
                    ax[i].set_xticks([])
                    ax[i].set_xticklabels([])
                    ax[i].imshow(x[0,i], vmin=np.min(x[0]), vmax=np.max(x[0]), cmap=plt.cm.binary)
        else:
            if filter_length == 1:
                for j in range(n_filter):
                    ax[j].set_ylabel('motif '+str(j))
                    ax[j].set_xlabel('frame 0')
                    ax[j].set_yticks([])
                    ax[j].set_yticklabels([])
                    ax[j].set_xticks([])
                    ax[j].set_xticklabels([])
                    ax[j].imshow(x[j,0], vmin=np.min(x[j]), vmax=np.max(x[j]), cmap=plt.cm.binary)
                
            else: 
                for j in range(n_filter):
                    ax[j,0].set_ylabel('motif '+str(j))
                    for i in range(filter_length):
                        ax[j,i].set_xlabel('frame '+str(i))
                        ax[j,i].set_yticks([])
                        ax[j,i].set_yticklabels([])
                        ax[j,i].set_xticks([])
                        ax[j,i].set_xticklabels([])
                        ax[j,i].imshow(x[j,i], vmin=np.min(x[j]), vmax=np.max(x[j]), cmap=plt.cm.binary)
                
        if self.eps == True:
            fig.savefig(self.folder+'motifs'+self.ending+'.eps',bbox_inches='tight')
        else:
            fig.savefig(self.folder+'motifs'+self.ending+'.png',bbox_inches='tight')
        
        plt.close('all')
        return()
    
    def plot_motifs_every_second_frame(self, x):
        n_filter = x.shape[0]
        filter_length = x.shape[1]
        if filter_length == 1:
            self.plot_motifs(x)
            return()
            
        if filter_length < 50:        
            width = filter_length/2.*7.
            height = n_filter*7.
        else:
            width = filter_length/2.*5.
            height = n_filter*5.
            
        params = {
                'axes.labelsize': 50,
                'font.size': 50,
                'legend.fontsize': 30,
                'xtick.labelsize': 20,
                'ytick.labelsize': 20,
                "text.usetex": False,
                'figure.figsize': [width, height]
                }
            
        plt.rcParams.update(params)
                       
        fig, ax = plt.subplots(n_filter,filter_length//2)
        if n_filter == 1:
            ax[0].set_ylabel('motif 0')
            for i, frame in enumerate(range(1, filter_length-1, 2)):
                ax[i].set_xlabel('frame '+str(frame))
                ax[i].set_yticks([])
                ax[i].set_yticklabels([])
                ax[i].set_xticks([])
                ax[i].set_xticklabels([])
                ax[i].imshow(x[0,frame], vmin=np.min(x[0]), vmax=np.max(x[0]), cmap=plt.cm.binary)
        else:
            for j in range(n_filter):
                ax[j,0].set_ylabel('motif '+str(j))
                for i, frame in enumerate(range(1, filter_length-1, 2)):
                    ax[j,i].set_xlabel('frame '+str(frame))
                    ax[j,i].set_yticks([])
                    ax[j,i].set_yticklabels([])
                    ax[j,i].set_xticks([])
                    ax[j,i].set_xticklabels([])
                    ax[j,i].imshow(x[j,frame], vmin=np.min(x[j]), vmax=np.max(x[j]), cmap=plt.cm.binary)
            
        if self.eps == True:
            fig.savefig(self.folder+'motifs_every_second'+self.ending+'.eps',bbox_inches='tight')
        else:
            fig.savefig(self.folder+'motifs_every_second'+self.ending+'.png',bbox_inches='tight')
        
        plt.close('all')
        return()
    
    def plot_motif_highlights(self,x,f,frames):
        highlights = x[f,frames]
        
        n_filter = 1
        filter_length = len(frames)
            
        if filter_length < 50:        
            width = filter_length*7.
            height = n_filter*7.
        else:
            width = filter_length*5.
            height = n_filter*5.
            
        params = {
                'axes.labelsize': 50,
                'font.size': 50,
                'legend.fontsize': 30,
                'xtick.labelsize': 20,
                'ytick.labelsize': 20,
                "text.usetex": False,
                'figure.figsize': [width, height]
                }
            
        plt.rcParams.update(params)
                       
        fig, ax = plt.subplots(1,filter_length)
        if filter_length == 1:
            ax.set_ylabel('motif '+str(f))
            ax.set_xlabel('frame '+str(frames[0]))
            ax.set_yticks([])
            ax.set_yticklabels([])
            ax.set_xticks([])
            ax.set_xticklabels([])
            ax.imshow(highlights[0], vmin=np.min(highlights), vmax=np.max(highlights), cmap=plt.cm.binary)
        
        else:
            ax[0].set_ylabel('motif '+str(f))
            for i in range(filter_length):
                ax[i].set_xlabel('frame '+str(frames[i]))
                ax[i].set_yticks([])
                ax[i].set_yticklabels([])
                ax[i].set_xticks([])
                ax[i].set_xticklabels([])
                ax[i].imshow(highlights[i], vmin=np.min(highlights), vmax=np.max(highlights), cmap=plt.cm.binary)
            
        if self.eps == True:
            fig.savefig(self.folder+'motif_'+str(f)+'_highlights'+self.ending+'.eps',bbox_inches='tight')
        else:
            fig.savefig(self.folder+'motif_'+str(f)+'_highlights'+self.ending+'.png',bbox_inches='tight')
        
        plt.close('all')
        return()
    
    def plot_motifs_difference_to_synch(self,motifs):
        n_filter = motifs.shape[0]
        filter_length = motifs.shape[1]
        if filter_length < 50:        
            width = filter_length*7.
            height = n_filter*7.
        else:
            width = filter_length*5.
            height = n_filter*5.
            
        params = {
                'axes.labelsize': 50,
                'font.size': 50,
                'legend.fontsize': 30,
                'xtick.labelsize': 20,
                'ytick.labelsize': 20,
                "text.usetex": False,
                'figure.figsize': [width, height]
                }
            
        plt.rcParams.update(params)
        
        synch_firing = np.zeros(motifs.shape)
        for m in range(n_filter):
            max_proj = np.max(motifs[m],axis=0)
            normed_max_proj = max_proj / np.max(max_proj)
            for f in range(filter_length):
                max_int = np.max(motifs[m,f])
                synch_firing[m,f] = normed_max_proj * max_int
        
        difference = motifs - synch_firing        
        
        fig, ax = plt.subplots(n_filter,filter_length)
        
        if n_filter == 1:
            if filter_length == 1:
                ax.set_ylabel('motif 0')
                ax.set_xlabel('frame 0')
                    
                ax.set_yticks([])
                ax.set_yticklabels([])
                ax.set_xticks([])
                ax.set_xticklabels([])
                
                limit = max(np.abs(np.max(difference[0,0])),np.abs(np.min(difference[0,0])))
                ax.imshow(difference[0,0], vmin=-limit, vmax=limit, cmap=plt.cm.RdBu)
            else:
                ax[0].set_ylabel('motif 0')
                for i in range(filter_length):
                    ax[i].set_xlabel('frame '+str(i))
                        
                    ax[i].set_yticks([])
                    ax[i].set_yticklabels([])
                    ax[i].set_xticks([])
                    ax[i].set_xticklabels([])
                    
                    limit = max(np.abs(np.max(difference[0,i])),np.abs(np.min(difference[0,i])))
                    ax[i].imshow(difference[0,i], vmin=-limit, vmax=limit, cmap=plt.cm.RdBu)
        else:
            if filter_Length == 1:
                for j in range(n_filter):
                    ax[j].set_ylabel('motif '+str(j))
                    if j == n_filter-1:
                        ax[j].set_xlabel('frame 0')
                        
                    ax[j].set_yticks([])
                    ax[j].set_yticklabels([])
                    ax[j].set_xticks([])
                    ax[j].set_xticklabels([])
                    
                    limit = max(np.abs(np.max(difference[j,0])),np.abs(np.min(difference[j,0])))
                    ax[j].imshow(difference[j,0], vmin=-limit, vmax=limit, cmap=plt.cm.RdBu)
            
            else:
                for j in range(n_filter):
                    ax[j,0].set_ylabel('motif '+str(j))
                    for i in range(filter_length):
                        if j == n_filter-1:
                            ax[j,i].set_xlabel('frame '+str(i))
                            
                        ax[j,i].set_yticks([])
                        ax[j,i].set_yticklabels([])
                        ax[j,i].set_xticks([])
                        ax[j,i].set_xticklabels([])
                        
                        limit = max(np.abs(np.max(difference[j,i])),np.abs(np.min(difference[j,i])))
                        ax[j,i].imshow(difference[j,i], vmin=-limit, vmax=limit, cmap=plt.cm.RdBu)
                        
        if self.eps == True:
            fig.savefig(self.folder+'motifs_difference'+self.ending+'.eps',bbox_inches='tight')
        else:
            fig.savefig(self.folder+'motifs_difference'+self.ending+'.png',bbox_inches='tight')
        
        plt.close('all')
        return()

    def plot_z(self,z):
        n_filter = z.shape[0]
        n_frames = z.shape[1]
        height = n_filter*5.
        width = n_frames/100.
        params = {
                'axes.labelsize': 50,
                'font.size': 50,
                'legend.fontsize': 30,
                'xtick.labelsize': 20,
                'ytick.labelsize': 20,
                "text.usetex": False,
                'figure.figsize': [width, height]
                }
            
        plt.rcParams.update(params)
        
        fig, ax = plt.subplots(n_filter)
        if n_filter > 1:    
            for filter_ in range(n_filter):
                ax[filter_].stem(np.arange(n_frames),z[filter_,:], 'r-', markerfmt='ro')
                ax[filter_].set_ylabel('activation\nmotif '+str(filter_))
                ax[filter_].set_xlim(0,n_frames)
                ax[filter_].set_ylim(0)
                if filter_ == n_filter -1:
                    ax[filter_].set_xlabel('frame')
        else:
            ax.stem(np.arange(n_frames),z[0,:], 'r-', markerfmt='ro')
            ax.set_ylabel('activation\nmotif 0')
            ax.set_xlim(0,n_frames)
            ax.set_ylim(0)
            ax.set_xlabel('frame')
        
        
            
        if self.eps == True:
            fig.savefig(self.folder+'z'+self.ending+'.eps',bbox_inches='tight')
        else:
            fig.savefig(self.folder+'z'+self.ending+'.png',bbox_inches='tight')
        
        plt.close('all')
        return()
    
    def plot_z_thresholded(self,z,thr=.7):
        thresholded = np.zeros(z.shape)
        for filter_ in range(z.shape[0]):
            max_ = np.max(z[filter_])
            thresholded[filter_,np.where(z[filter_]>thr*max_)] = z[filter_,np.where(z[filter_]>thr*max_)]
        
        self.ending = '_thresholded' + self.ending
        self.plot_z(thresholded)
        self.ending = self.ending[len('_thresholded'):]
        
        plt.close('all')
        return()
        
    def plot_z_zoomed(self, z, start_stop=None):
        n_filter = z.shape[0]
        n_frames = z.shape[1]
        height = n_filter*5.
        width = n_frames/100. + 0.25*n_frames/100.
        
        params = {
                'axes.labelsize': 50,
                'font.size': 50,
                'legend.fontsize': 30,
                'xtick.labelsize': 20,
                'ytick.labelsize': 20,
                "text.usetex": False,
                'figure.figsize': [width, height]
                }
            
        plt.rcParams.update(params)
        
        
        fig, ax = plt.subplots(n_filter, 2, sharey=True, gridspec_kw={'width_ratios': [4, 1]},)
        if n_filter > 1:    
            for j in range(2):
                for filter_ in range(n_filter):
                    if j == 0:
                        ax[filter_, j].stem(np.arange(n_frames),z[filter_,:], 'r-', markerfmt='ro')
                        ax[filter_, j].set_xlim(0,n_frames)
                        ax[filter_, j].set_ylim(0,np.max(z[filter_])*1.1)
                        ax[filter_, j].set_ylabel('activation\nmotif '+str(filter_))
                        if filter_ == n_filter -1:
                            ax[filter_,j].set_xlabel('frame')
                    else:
                        ax[filter_, j].stem(np.arange(start_stop[filter_][0], start_stop[filter_][1]),z[filter_,start_stop[filter_][0]:start_stop[filter_][1]], 'r-', markerfmt='ro',  mew=4)
                        ax[filter_, j].xaxis.set_ticks(np.arange(start_stop[filter_][0], start_stop[filter_][1],10)[1:])
                        ax[filter_, j].set_ylim(0,np.max(z[filter_])*1.1)
        
        else:  
            for j in range(2):
                if j == 0:
                    ax[j].stem(np.arange(n_frames),z[0,:], 'r-', markerfmt='ro')
                    ax[j].set_xlim(0,n_frames)
                    ax[j].set_ylim(0,np.max(z[0])*1.1)
                    ax[j].set_ylabel('activation\nmotif 0')
                    ax[j].set_xlabel('frame')
                else:
                    ax[j].stem(np.arange(start_stop[0][0], start_stop[0][1]),z[0,start_stop[0][0]:start_stop[0][1]], 'r-', markerfmt='ro',  mew=4)
                    ax[j].xaxis.set_ticks(np.arange(start_stop[0][0], start_stop[0][1],10)[1:])
                    ax[j].set_ylim(0,np.max(z[0])*1.1)
                      
        if self.eps == True:
            fig.savefig(self.folder+'z_zoomed'+self.ending+'.eps',bbox_inches='tight')
        else:
            fig.savefig(self.folder+'z_zoomed'+self.ending+'.png',bbox_inches='tight')
        return()

    def make_motif_movies(self,x):
        n_filter = x.shape[0]
        
        for n in range(n_filter):
            a = x[n]
            a = ((a - a.min())/(a.max() - a.min()) * 255).astype(np.uint8)
            imlist = []
            for m in a:
                imlist.append(Image.fromarray(m))
            
            # saving it as a GIF and then opening it and saving it as a TIFF
            imlist[0].save(self.folder+"motif_"+str(n)+self.ending+".gif", save_all=True, append_images=imlist[1:])
            Image.open(self.folder+"motif_"+str(n)+self.ending+".gif").save(folder+"motif_"+str(n)+ending+".tif", save_all=True)
            
        
        return()
    

