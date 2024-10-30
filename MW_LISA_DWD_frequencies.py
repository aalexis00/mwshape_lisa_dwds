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
    'font.size': 10,                # Set global font size
    'font.family': 'serif',
    'figure.figsize': [8, 6],       # Set global figure size
    'axes.titlesize': 12,           # Set title size
    'axes.labelsize': 10,           # Set axis label size
    'lines.linewidth': 2,           # Set line width for plots
    'xtick.major.pad': '5.0',       # Padding for major x-ticks
    'xtick.major.size': '3.0',      # Size for major x-ticks
    'xtick.major.width': '1.0',     # Width for major x-ticks
    'xtick.minor.pad': '5.0',       # Padding for minor x-ticks
    'xtick.minor.size': '1.5',      # Size for minor x-ticks
    'xtick.minor.width': '1.0',     # Width for minor x-ticks
    'ytick.major.pad': '5.0',       # Padding for major y-ticks
    'ytick.major.size': '3.0',      # Size for major y-ticks
    'ytick.major.width': '1.0',     # Width for major y-ticks
    'ytick.minor.pad': '5.0',       # Padding for minor y-ticks
    'ytick.minor.size': '1.5',      # Size for minor y-ticks
    'ytick.minor.width': '1.0',     # Width for minor y-ticks
    'axes.titlepad': '10.0',        # Padding for the title
    'axes.labelpad': '10.0',        # Padding for axis labels
    'figure.autolayout': False,     # Disable tight_layout() globally
    'figure.constrained_layout.use': True,  # Use constrained layout
    'figure.constrained_layout.w_pad': 1.5, # Width padding
    'figure.constrained_layout.h_pad': 1.5, # Height padding
})


def get_freq(dat, freq_list):
    
    dat_freq = 2 * dat['f_orb_f'].values
    
    return np.concatenate((dat_freq, freq_list))


'''Setting parameters'''
linestyles = ['solid', 'dashed']
components_fiducial = ['halo', 'bulge', 'thick disk', 'thin disk']
components_empirical = ['GSE halo', 'in situ halo', 'bulge', 'thick disk', 'thin disk']
components_model = (['halo', 'bulge', 'thick disk', 'thin disk'],
                    ['GSE halo', 'in situ halo', 'bulge', 'thick disk', 'thin disk'])

#model_colors = [['green', 'blue', 'orange', 'red'], ['cadetblue', 'lime', 'mediumpurple', 'peru', 'lightcoral']]
models = ['fiducial', 'empirical']
DWD_types_colors = ['pink', 'violet', 'paleturquoise', 'brown']
DWD_types = ["HeHe", "HeCo", "CoCo", "ONeX"]
    
def DWD_types_freq(models, DWD_types_colors, DWD_types, linestyles):  
    
    '''Make a frequency distribution plot for each DWD types, for each model'''

    fig, axes = plt.subplots(len(models), 1, figsize=(4,6))
    
    for m, components, ax, ls in zip(models, components_model, axes, linestyles):
        
        for DWD, color in zip(DWD_types, DWD_types_colors):

            DWD_frequency = []
            
            for component in components:

                dat = pd.read_hdf(f"{component.replace(' ', '_')}_double_white_dwarfs_{m}.h5", 
                                key=f"{DWD}_{component.replace(' ', '_')}", mode ='r', errors='strict')
                
                if len(DWD_frequency) > 0:
                    DWD_frequency  = get_freq(dat, DWD_frequency ) 
                else:
                    DWD_frequency = 2 * dat['f_orb_f'].values

            ax.hist(np.log10(DWD_frequency), label=DWD, edgecolor=color, histtype='step',
                        bins= np.linspace(-4, -0.5, 25),ls=ls)
            
            #print(m + ' ' + str(DWD) + ' ' + str(len(DWD_frequency)))

            ax.set_ylabel('# of Resolved Sources', fontsize=10)
            ax.set_xlabel(r"$\log_{10}(f_{\rm{GW}}/\rm{Hz})$", fontsize=10)
            ax.minorticks_on()

            ax.set_title(f"{m.title()} Model", fontsize=10)

            ax.set_yscale('log')
            ax.legend(loc='upper right', fontsize='small', prop={'size':7.5})
    

    plt.tight_layout()  # Adjust layout to prevent overlap


    fig.savefig("DWD_frequency.png")
            
       

DWD_types_freq(models, DWD_types_colors, DWD_types, linestyles)