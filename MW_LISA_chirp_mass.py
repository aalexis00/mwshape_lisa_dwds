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



def get_chirp_mass(dat, chirp_mass_list):
    
    dat_chirp_mass = dat['chirp mass'].values
    
    return np.concatenate((dat_chirp_mass, chirp_mass_list))


'''Setting parameters'''
linestyles = ['solid', 'dashed']
#components_fiducial = ['halo', 'thick disk']
#components_empirical = ['GSE halo', 'in situ halo', 'thick disk']
components_model = (['halo', 'thick disk'],
                    ['GSE halo', 'in situ halo', 'thick disk'])
model_colors = [['green', 'orange'], ['cadetblue', 'lime', 'lightcoral']]
models = ['fiducial', 'empirical']
DWD_types = ["HeHe", "HeCo", "CoCo", "ONeX"]
    
def chirp_mass_plot(components_model, components_colors, models, DWD_types, linestyles):  
    
    '''Make a chirp mass distribution plot for each component, for each model'''

    plt.figure()


    for m, components, ls, component_colors in zip(models, components_model, linestyles, model_colors):
        for component, c,  in zip(components, component_colors):
            chirp_mass =[]
            
            for t in DWD_types:
                dat = pd.read_hdf(f"{component.replace(' ', '_')}_double_white_dwarfs_{m}.h5", 
                                  key=f"{t}_{component.replace(' ', '_')}", mode ='r', errors='strict')
                if len(chirp_mass) > 0:
                    chirp_mass = get_chirp_mass(dat, chirp_mass) 
                else:
                    chirp_mass = dat['chirp mass'].values
            
            #labelname = word if word.upper() == word else word.title() for word in component
           
            label_name = ' '.join(word if word.upper() == word else word.title() for word in component.split())
        
            plt.hist(chirp_mass, label=label_name, edgecolor=c, histtype='step',
                    bins= np.linspace(0.1, 1.25, 25),ls=ls)
        
        #ax.tick_params(axis='both', which='major', labelsize=10)
        plt.yscale('log')

        plt.legend(loc='upper right', fontsize='small', prop={'size':7.5})
        
    plt.ylabel('# of Resolved Sources', fontweight='bold')
    plt.xlabel('Chirp Masses', fontweight='bold')
    plt.tight_layout()

    plt.savefig("chirp_masses.png")
            
       

chirp_mass_plot(components_model, model_colors, models, DWD_types, linestyles)