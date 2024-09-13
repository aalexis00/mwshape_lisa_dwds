import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib.colors import LinearSegmentedColormap

import h5py
import pandas as pd



def get_dist(dat, dist_list):
    
    dat_dist = dat['dist'].values
    
    return np.concatenate((dat_dist, dist_list))


'''Setting parameters'''
linestyles = ['solid', 'dashed']
components_fiducial = ['halo', 'bulge', 'thick disk', 'thin disk']
components_empirical = ['GSE halo', 'in situ halo', 'bulge', 'thick disk', 'thin disk']
components_model = (['halo', 'bulge', 'thick disk', 'thin disk'],
                    ['GSE halo', 'in situ halo', 'bulge', 'thick disk', 'thin disk'])
components_colors = ['green', 'blue', 'orange', 'red']
models = ['fiducial', 'empirical']
DWD_types = ["HeHe", "HeCo", "CoCo", "ONeX"]
    
def distance_plot(components_model, components_colors, models, DWD_types, linestyles):  
    
    '''Make a distance distribution plot for each DWD type, for each model'''

    fig, axes = plt.subplots(len(models), 1, figsize=(4,6))
    

    for m, components, ax, ls in zip(models, components_model, axes, linestyles):
        for component, c,  in zip(components, components_colors):
            dist =[]
            
            for t in DWD_types:
                dat = pd.read_hdf(f"{component.replace(' ', '_')}_double_white_dwarfs_{m}.h5", 
                                  key=f"{t}_{component.replace(' ', '_')}", mode ='r', errors='strict')
                if len(dist) > 0:
                    dist = get_dist(dat, dist) 
                else:
                    dist = dat['dist'].values
            
            ax.hist(dist, label=component.title(), edgecolor=c, histtype='step',
                    bins= np.linspace(0, 50, 25),ls=ls)
        
        ax.set_xlabel('Distances (kpc)', fontsize=10)
        ax.set_ylabel('Number of Resolved Sources', fontsize=10)
        ax.tick_params(axis='both', which='major', labelsize=10)
        ax.set_title(f"{m.title()} Model")
        ax.set_yscale('log')

        ax.legend(loc='upper right', fontsize='small')
        plt.tight_layout()

        fig.savefig("distances.png")
            
   
    
    #fig.savefig("distances.png")
    

distance_plot(components_model, components_colors, models, DWD_types, linestyles)