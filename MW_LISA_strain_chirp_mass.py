import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib.colors import LinearSegmentedColormap

import h5py
import pandas as pd
import seaborn as sns
import math

# re-defining plotting defaults
from matplotlib import rcParams
# Update multiple rcParams settings at once using rcParams

rcParams.update({
    'font.size': 10,                # Set global font size
    'font.family': 'serif',
    'figure.figsize': [8, 6],       # Set global figure size
    'axes.titlesize': 10,           # Set title size
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

'''Setting parameters'''
linestyles = ['solid', 'dashed']
all_components = ['halo', 'bulge', 'thick disk', 'thin disk',
                  'GSE halo', 'in situ halo', 'bulge', 'thick disk', 'thin disk']
components_model = (['halo', 'bulge', 'thick disk', 'thin disk'],
                    ['GSE halo', 'in situ halo', 'bulge', 'thick disk', 'thin disk'])
model_colors = [['green', 'blue', 'orange', 'red'], ['cadetblue', 'lime', 'mediumpurple', 'peru', 'lightcoral']]
models = ['fiducial', 'empirical']
DWD_types = ["HeHe", "HeCo", "CoCo", "ONeX"]


def get_chirp_mass(dat, chirp_mass_list):
    
    dat_chirp_mass = dat['chirp mass'].values
    
    return np.concatenate((dat_chirp_mass, chirp_mass_list))

def get_strain(dat, strain_list):
    
    dat_strain = dat['strain'].values
    
    return np.concatenate((dat_strain, strain_list))


def strain_chirp_mass_plot(all_components, components_model, model_colors, models, DWD_types, linestyles, ncols=4):  
    # Calculate the number of components and set up rows and columns for subplots
    num_components = sum(len(components) for components in components_model)

    ncols = int(np.sqrt(num_components))
    nrows = int(num_components / ncols)  # Calculate rows needed for the given number of columns

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(ncols * 4, nrows * 3))
    
    axes = axes.flatten()  # Flatten to handle as a 1D array for easy iteration

    plot_index = 0

    # Iterate over models and components
    for m, components, ls, component_colors in zip(models, components_model, linestyles, model_colors):

        for component, ax, c in zip(components, axes, component_colors):

            ax = axes[plot_index]  # Get the current subplot

            plot_index += 1

            chirp_mass = []
            strain = []
            
            for t in DWD_types:
                # Load data
                dat = pd.read_hdf(f"{component.replace(' ', '_')}_double_white_dwarfs_{m}.h5", 
                                  key=f"{t}_{component.replace(' ', '_')}", mode='r')
                
                # Get frequency and chirp mass values
                if len(chirp_mass) > 0:
                    chirp_mass = get_chirp_mass(dat, chirp_mass) 
                else:
                    chirp_mass = dat['chirp mass'].values

                if len(strain) > 0:
                    strain = get_strain(dat, strain) 
                else:
                    strain = dat['strain'].values

            
            # Create scatter plot
            label_name = ' '.join(word if word.upper() == word else word.title() for word in component.split())
            ax.scatter(np.log10(chirp_mass), np.log10(strain), label=label_name, color=c, s=1, alpha=0.5)

            # Set titles, labels, and legends
            ax.set_xlabel(r"Chirp Masses $[\rm{M}_{\odot}]$")
            ax.set_ylabel(r"Strain")
            ax.set_title(f"{m.title()} Model - {label_name}")
            ax.legend(loc='upper right', fontsize=7.5)


    # Adjust layout to prevent overlap and save the figure
    plt.tight_layout()
    fig.savefig("strain_chirp_mass_components.png")
            

strain_chirp_mass_plot(all_components, components_model, model_colors, models, DWD_types, linestyles)