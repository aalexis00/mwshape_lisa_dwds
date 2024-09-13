import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib.colors import LinearSegmentedColormap

import h5py
import pandas as pd

def get_positions(dat, i_pos, i_pos_list):
    
    dat_i_pos = dat[i_pos].values
    
    return np.concatenate((dat_i_pos, i_pos_list))

'''Setting parameters'''
linestyles = ['solid', 'dashed']
components_fiducial = ['bulge', 'thick disk', 'thin disk']
components_empirical = ['in situ halo', 'bulge', 'thick disk', 'thin disk']
components_model = (['halo', 'bulge', 'thick disk', 'thin disk'],['GSE halo', 'in situ halo', 'bulge', 'thick disk', 'thin disk'])
models = ['fiducial', 'empirical']
DWD_types = ["HeHe", "HeCo", "CoCo", "ONeX"]


def contour_plot_2D(components, models, DWD_types, linestyles):
    '''Creating a 2D (x-y positions) contour plot of all components, as compared to fiducial halo and GSE halo'''    

    fig, axes = plt.subplots(len(models), 1, figsize=(6,12))


    for m, components, ax, ls in zip(models, components_model, axes, linestyles):
        
        background_components = []
        
        x_pos = []
        y_pos = []
        
        x_pos_o_halo = []
        y_pos_o_halo = []
        x_pos_SNR_halo_ = []
        y_pos_SNR_halo = []
        
        for component in components:
            
            if component == 'halo' or component == 'GSE halo':
                
                for t in DWD_types:
                    
                    dat_h_o = pd.read_hdf(f"{component.replace(' ', '_')}_original_double_white_dwarfs_{m}.h5", 
                                  key=f"{t}_{component.replace(' ', '_')}", mode ='r', errors='strict')
                    
                    dat_h_SNR = pd.read_hdf(f"{component.replace(' ', '_')}_double_white_dwarfs_{m}.h5", 
                                  key=f"{t}_{component.replace(' ', '_')}", mode ='r', errors='strict')
                    
                    
                    if len(x_pos_o_halo) > 0:
                        x_pos_o_halo  = get_positions(dat_h_o, 'x_pos', x_pos_o_halo)
                        y_pos_o_halo  = get_positions(dat_h_o, 'y_pos', y_pos_o_halo)
                        
                        x_pos_SNR_halo = get_positions(dat_h_SNR, 'x_pos',x_pos_SNR_halo)
                        y_pos_SNR_halo  = get_positions(dat_h_SNR, 'y_pos', y_pos_SNR_halo)
                    
                    else:
                        x_pos_o_halo  = dat_h_o['x_pos'].values
                        y_pos_o_halo  = dat_h_o['y_pos'].values
                        
                        x_pos_SNR_halo  = dat_h_SNR['x_pos'].values
                        y_pos_SNR_halo  = dat_h_SNR['y_pos'].values
                        
                
            else:
                
                background_components.append(component)
                
                for t in DWD_types:
                    
                    dat = pd.read_hdf(f"{component.replace(' ', '_')}_original_double_white_dwarfs_{m}.h5", 
                                  key=f"{t}_{component.replace(' ', '_')}", mode ='r', errors='strict')
                
            
                    if len(x_pos) > 0:
                        x_pos = get_positions(dat, 'x_pos', x_pos)
                        y_pos = get_positions(dat, 'y_pos', y_pos)
                    
                    else:
                        x_pos = dat['x_pos'].values
                        y_pos = dat['y_pos'].values
        
        counts, xedges, yedges = np.histogram2d(x_pos + 8, y_pos, bins= 60)
        
        contour1 = ax.contour(np.log10(counts.T), extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],
                   linewidths=2, cmap='Greys', alpha=0.8, levels = 5)
        
        counts_halo_o, xedges_halo_o, yedges_halo_o = np.histogram2d(np.array(x_pos_o_halo) + 8, y_pos_o_halo, bins= 60)
        
        print(f"x_pos_o_halo: {x_pos_o_halo}")
        print(f"y_pos_o_halo: {y_pos_o_halo}")
        contour2 = ax.contour(np.log10(counts_halo_o.T), extent=[xedges_halo_o[0], xedges_halo_o[-1], 
                                                                 yedges_halo_o[0], yedges_halo_o[-1]],
                                linewidths=2, cmap='Greens', alpha=0.8, label = 'Halo', levels = 5)
        
        ax.scatter(np.array(x_pos_SNR_halo) + 8, y_pos_SNR_halo , s=1, color='green')
        
        
        # Create proxy artists for the contours to add them to the legend
        from matplotlib.lines import Line2D
        
        label1 = ' + '.join([comp.title() for comp in background_components])
        proxy1 = Line2D([0], [0], color='grey', lw=2, label = label1)
        
        if m == 'fiducial':
            proxy2 = Line2D([0], [0], color='green', lw=2, label = 'Halo' )
        elif m == 'empirical':
            proxy2 = Line2D([0], [0], color='green', lw=2, label = 'GSE Halo')
        
        ax.set_aspect('equal', adjustable='box') 
        
        # Set labels, limits, ticks, and add legend with proxy artists
        ax.set_xlabel('Galactocentric X [kpc]', fontsize=10)
        ax.set_ylabel('Galactocentric Y [kpc]', fontsize=10)
        ax.set_xlim(-40, 40)
        ax.set_ylim(-40, 40)
        ax.tick_params(axis='x', labelsize=10)
        ax.tick_params(axis='y', labelsize=10)

        # Add custom legend with scatter and proxy artists for contours
        resolved_label = 'Resolved ' + proxy2.get_label() + ' Sources'
        ax.legend(handles=[proxy1, proxy2, ax.scatter([], [], s=1, color='green', label=resolved_label)],
          loc='upper right', fontsize=10)

        fig.tight_layout()

        fig.savefig("positions_contour.png")
        
contour_plot_2D(components_model, models, DWD_types, linestyles)
