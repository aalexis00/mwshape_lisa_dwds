import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib.colors import LinearSegmentedColormap

import h5py
import pandas as pd



def get_freq(dat, freq_list):
    
    dat_freq = 2 * dat['f_orb_f'].values
    
    return np.concatenate((dat_freq, freq_list))


'''Setting parameters'''
linestyles = ['solid', 'dashed']
components_fiducial = ['halo', 'bulge', 'thick disk', 'thin disk']
components_empirical = ['GSE halo', 'in situ halo', 'bulge', 'thick disk', 'thin disk']
components_model = (['halo', 'bulge', 'thick disk', 'thin disk'],
                    ['GSE halo', 'in situ halo', 'bulge', 'thick disk', 'thin disk'])
components_colors = ['green', 'blue', 'orange', 'red']
models = ['fiducial', 'empirical']
DWD_types = ["HeHe", "HeCo", "CoCo", "ONeX"]
    
def frequency_plot(components_model, components_colors, models, DWD_types, linestyles):  
    
    '''Make a GW frequency distribution plot for each DWD type, for each model'''

    fig, axes = plt.subplots(len(models), 1, figsize=(4,6))
    

    for m, components, ax, ls in zip(models, components_model, axes, linestyles):
        for component, c,  in zip(components, components_colors):
            freq =[]
            
            for t in DWD_types:
                dat = pd.read_hdf(f"{component.replace(' ', '_')}_double_white_dwarfs_{m}.h5", 
                                  key=f"{t}_{component.replace(' ', '_')}", mode ='r',
                                  errors='strict')
                if len(freq) > 0:
                    freq = get_freq(dat, freq) 
                else:
                    freq = 2 * dat['f_orb_f'].values
            
            ax.hist(np.log10(freq), label=component.title(), edgecolor=c, histtype='step',
                    bins= np.linspace(-4, -0.5, 25),ls=ls)
        
        ax.set_xlabel('Gravitational Wave Frequencies (log\u2081\u2080)', fontsize=10)
        ax.set_ylabel('Number of Resolved Sources', fontsize=10)
        ax.tick_params(axis='both', which='major', labelsize=10)
        ax.set_title(f"{m.title()} Model")
        ax.set_yscale('log')

        ax.legend(loc='upper right', fontsize='small')
        plt.tight_layout()

        fig.savefig("frequencies.png")
            

frequency_plot(components_model, components_colors, models, DWD_types, linestyles)