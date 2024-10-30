import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib.colors import LinearSegmentedColormap

import h5py
import pandas as pd

# re-defining plotting defaults
from matplotlib import rcParams
# Update multiple rcParams settings at once using rcParams
rcParams.update({
    'font.size': 12,                # Set global font size
    'font.family': 'serif',
    'figure.figsize': [8, 6],       # Set global figure size
    'axes.titlesize': 13,           # Set title size
    'axes.labelsize': 10,           # Set axis label size
    'lines.linewidth': 2,            # Set line width for plots
    'xtick.major.pad': '5.0',
    'xtick.major.size': '5.5',
    'xtick.major.width': '1.0',
    'xtick.minor.pad': '5.0',
    'xtick.minor.size': '2.5',
    'xtick.minor.width': '1.0',
    'ytick.major.pad': '5.0',
    'ytick.major.size': '5.5',
    'ytick.major.width': '1.0',
    'ytick.minor.pad': '5.0',
    'ytick.minor.size': '2.5',
    'ytick.minor.width': '1.0',
    'axes.titlepad': '10.0',
    'axes.labelpad': '10.0',
})


def get_distance(dat, dist_list):
    
    dat_dist = dat['dist'].values
    
    return np.concatenate((dat_dist, dist_list))


'''Setting parameters'''
linestyles = ['solid', 'dashed']
components_fiducial = ['halo', 'bulge', 'thick disk', 'thin disk']
components_empirical = ['GSE halo', 'in situ halo', 'bulge', 'thick disk', 'thin disk']
components_model = (['halo', 'bulge', 'thick disk', 'thin disk'],
                    ['GSE halo', 'in situ halo', 'bulge', 'thick disk', 'thin disk'])
#model_colors = [['green', 'blue', 'orange', 'red'], ['cadetblue', 'lime', 'mediumpurple', 'peru', 'lightcoral']]
models = ['fiducial', 'empirical']
DWD_types_colors = ['pink', 'violet', 'paleturquoise', 'yellow']
DWD_types = ["HeHe", "HeCo", "CoCo", "ONeX"]
    
def DWD_types_distance(models, DWD_types_colors, DWD_types, linestyles):  
    
    '''Make a distance distribution plot for each DWD types, for each model'''

    fig, axes = plt.subplots(len(models), 1, figsize=(4,6))
    
    for m, components, ax, ls in zip(models, components_model, axes, linestyles):
        
        for DWD, color in zip(DWD_types, DWD_types_colors):

            DWD_distance = []
            
            for component in components:

                dat = pd.read_hdf(f"{component.replace(' ', '_')}_double_white_dwarfs_{m}.h5", 
                                key=f"{DWD}_{component.replace(' ', '_')}", mode ='r', errors='strict')
                
                if len(DWD_distance) > 0:
                    DWD_distance = get_distance(dat, DWD_distance ) 
                else:
                    DWD_distance = dat['dist'].values
                                    
            ax.hist(np.log10(DWD_distance), label=DWD, edgecolor=color, histtype='step', ls=ls)
            
            #ax.set_ylabel('# of Resolved Sources', fontsize=10)
            #ax.tick_params(axis='both', which='major', labelsize=10)
            ax.set_title(f"{m.title()} Model", fontsize=10)
            ax.set_yscale('log')


            ax.legend(loc='upper right', fontsize='small', prop={'size':7.5})
            
    
     # Global x and y labels
    fig.text(0.5, 0.02, 'Distances [kpc]', ha='center', fontsize=10, fontweight='bold')  # x-label
    fig.text(0.02, 0.5, '# of Resolved Sources', va='center', rotation='vertical', fontsize=10, fontweight='bold')  # y-label

    plt.tight_layout(pad=3.0)  # Adjust layout to prevent overlap

    fig.savefig("DWD_distance.png")
            
       

DWD_types_distance(models, DWD_types_colors, DWD_types, linestyles)