import legwork

import legwork.source as source
import legwork.visualisation as vis

from legwork import evol, utils
from legwork import snr

import numpy as np
import astropy.units as u
import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib.colors import LinearSegmentedColormap

from astropy.visualization import quantity_support
quantity_support()

import h5py
import pandas as pd


def get_dist(dat, dist_list):
    
    dat_dist = dat['dist'].values
    
    return np.concatenate((dat_dist, dist_list))


'''Setting parameters'''
linestyles = ['solid', 'dashed']
components_fiducial = ['halo']
components_empirical = ['GSE halo', 'in situ halo']
components_model = (['halo'],['GSE halo', 'in situ halo'])
components_colors = ['green', 'darkseagreen']
models = ['fiducial', 'empirical']
DWD_types = ["HeHe", "HeCo", "CoCo", "ONeX"]
    
def distance_comparison_plot(components, components_colors, models, DWD_types, linestyles):
    '''Compare GSE Halo + In-situ Halo and Fiducial Halo Distance Distribution'''    

    plt.figure(figsize=(5,4))

    for m, components, ls in zip(models, components_model, linestyles):
        dist =[]
        
        for component, c in zip(components, components_colors):
           
            for t in DWD_types:
                dat = pd.read_hdf(f"{component.replace(' ', '_')}_double_white_dwarfs_{m}.h5", 
                                  key=f"{t}_{component.replace(' ', '_')}", mode ='r', errors='strict')
                if len(dist) > 0:
                    dist = get_dist(dat, dist) 
                else:
                    dist = dat['dist'].values
        
        label_name = ' + '.join(' '.join(word if word.upper() == word else word.title() 
                                         for word in label.split()) for label in components)
        
        plt.hist(dist, label=label_name, bins= np.linspace(0, 50, 25), alpha=0.7, rwidth=0.9, 
                 histtype='step', color=c, ls=ls)
            
        plt.xlabel('Distances (kpc)', fontsize=10)
        plt.ylabel('Number of Resolved Sources', fontsize=10)
        plt.xticks(fontsize=10)
        plt.yticks(fontsize=10)
        plt.title("Halo Comparison")
        plt.yscale('log')

        plt.legend(loc='upper right', fontsize='small')
        plt.savefig("distances_halo_comparison.png")

distance_comparison_plot(components_model, components_colors, models, DWD_types, linestyles)