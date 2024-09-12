import legwork.source as source
import legwork.visualisation as vis

from legwork import evol, utils
from legwork import snr

import numpy as np
import astropy.units as u
import matplotlib.pyplot as plt

from astropy.visualization import quantity_support
quantity_support()

import h5py
import pandas as pd

import argparse

from argparse import ArgumentParser

from datetime import datetime

'''
parser = argparse.ArgumentParser()

parser.usage = 'Use this to run the scripts for each component'

parser.add_argument("component", type = str, help= " Returns full dataset of DWD's in component with SNR > 7", 
                    choices = ['GSE halo', 'halo', 'bulge', 'thin disk', 'thick disk'])


args = parser.parse_args()
'''

'''File names for GSE halo'''
#metalliticity of 0.001, bin 6
hh_filename_6 = 'fiducial_10_10_6.h5' #He + He
coh_filename_6 = 'fiducial_11_10_6.h5' # CO + He
coco_filename_6 = 'fiducial_11_11_6.h5' # CO + CO
oho_filename_6 = 'fiducial_12_10_12_6.h5' #ONe + X


'''File names for in-situ halo and high-alpha disk (thick disk)'''
#metallicity bin 10, corresponding to 0.02
hh_filename_10 = 'fiducial_10_10_10.h5'
coh_filename_10 = 'fiducial_11_10_10.h5'
coco_filename_10 = 'fiducial_11_11_10.h5'
oho_filename_10 = 'fiducial_12_10_12_10.h5'


'''File names for low-alpha disk (thin disk) and bulge, same as in fiducial case'''

#metallicity bin 14, corresponding to 0.02
hh_filename_14 = 'fiducial_10_10_14.h5'
coh_filename_14 = 'fiducial_11_10_14.h5'
coco_filename_14 = 'fiducial_11_11_14.h5'
oho_filename_14 = 'fiducial_12_10_12_14.h5'

'''GSE Halo information'''
M_GSE_halo = 6 * 10**8 * u.Msun #from Han, 2022

age_GSE_halo = 10 * 1e9 * u.yr

'''In situ Halo information'''

M_fiducial_halo = 1.4 * 10**9 * u.Msun #from Deason, 2019

M_in_situ_halo = M_fiducial_halo - M_GSE_halo

age_in_situ_halo = 14 * 1e9 * u.yr #all formed in one burst according to Robin, 2003


'''Thin disk information''' #McMillan, 2011

cen_surf_dens_thin = 816.6
R_d_thin = 2900
M_disk_thin = 2*np.pi * cen_surf_dens_thin * R_d_thin**2 *u.Msun 


'''Thick disk information''' #McMillan, 2011

cen_surf_dens_thick = 209.5
R_d_thick = 3310
M_disk_thick = 2*np.pi * cen_surf_dens_thick * R_d_thick**2 *u.Msun
age_thick_disk = 11 * 1e9 * u.yr #all formed in one burst according to Robin, 2003

'''Bulge information''' #McMillan, 2011

M_bulge = 8.9 * 10**9 * u.Msun #McMillan, 2011

age_bulge = 10 * 1e9 * u.yr #all formed in one burst according to Robin, 2003


'''Evolution times'''

#Calculates all gravitational evolution times given the time elapsed since enterred ZAMS

#Calculates all gravitational evolution times given the time elapsed since enterred ZAMS
def t_grav_evol(component, timeelapsed, n): 
    if component == 'thin disk': #Thin disk binaries did not form in one burst
        birth = np.random.uniform(0, 10, n) * 1e9 * u.yr
        tgrav = birth - timeelapsed*u.Myr
    else:
        if component == 'thick disk':
            age_component =  age_thick_disk
        if component == 'bulge':
            age_component = age_bulge
        if component == 'in situ halo':
            age_component =  age_in_situ_halo
        if component == 'GSE halo':
            age_component = age_GSE_halo 
        birth = np.random.uniform(age_component.value - 1e9 , age_component.value,n)
        tgrav = birth* u.yr - timeelapsed*u.Myr
        
    return tgrav.value, birth


#Calculates all gravitational evolution times for given fiducial model
def findtimes(d, component):
    tg, age = t_grav_evol(component, d['tphys'].values, len(d))
    d["tgrav_ev"] = tg 
    d['age']= age
    
    d= d.loc[(d["tphys"] * 1e6 < d['age'])]
          
    return d

'''Binary evolution'''

'''
Function that creates a dictionary from given set DWD data with values interested in, 
selecting for binaries with gravitational evolution times less than their merger times 
'''
def select_DWDs(data):
    print('Original length of data is: ' + str(len(data)))
    
    #Select those w/ sep < 10000
    data = data.loc[data['sep'] < 5000] #changed from 10000
    
    print('Length of data with separation less than 5,000 is: ' + str(len(data)))

    
    #Retrieve relevant values from updated data
    m_1 = data['mass_1'].values * u.Msun
    m_2 = data['mass_2'].values * u.Msun
    porb = data['porb'].values
    f_orb_i = 1/(porb * u.day)
    forb_i = f_orb_i.to(u.Hz)
    #print(f_orb_i)         
    
    #Find merger (inspiral) time
    t_merge = evol.get_t_merge_circ(m_1=m_1, m_2=m_2, f_orb_i=f_orb_i)
    t_merge = t_merge.to(u.yr)
    #print(t_merge)
    
    #Update with values of gravitational evolution time less than t_merge array and less than age of binaries
    data = data.loc[ (data["tgrav_ev"] < t_merge.value) & (data["tgrav_ev"] < data['age'])] #account for thin disk
    print('Length of data with selected gravitational evolution time is: ' + str(len(data)))

    tg = data['tgrav_ev'].values * u.yr
    
    #Retrieve relevant values from updated data again
    m_1 = data['mass_1'].values * u.Msun
    m_2 = data['mass_2'].values * u.Msun
    porb = data['porb'].values
    f_orb_i = 1/(porb * u.day)
    f_orb_i = f_orb_i.to(u.Hz)
    #print(f_orb_i)
    
    nchunks = 100000
    
    m_1 = [m_1[i:i+nchunks] for i in range(0, len(m_1), nchunks)]
    m_2 = [m_2[i:i+nchunks] for i in range(0, len(m_2), nchunks)]
    porb = [porb[i:i+nchunks] for i in range(0, len(porb), nchunks)]
    f_orb_i = [f_orb_i[i:i+nchunks] for i in range(0, len(f_orb_i), nchunks)]
    tg = [tg[i:i+nchunks] for i in range(0, len(tg), nchunks)]
    
    f_orb_f = []
    for i in range(len(m_1)):
        
        new_m_1 = m_1[i]
        new_m_2 = m_2[i]
        new_porb= porb[i]
        new_f_orb_i = f_orb_i[i]
        new_tg = tg[i]
        
        #Evolve to find present day orbital frequency
        forb = evol.evol_circ(t_evol = new_tg, m_1=new_m_1, m_2=new_m_2, f_orb_i=new_f_orb_i, output_vars='f_orb')
        
        #print(len(forb))
        #print(forb)

        f_orb_f_v = forb[:,-1].value
        
        #import pdb
        
        #pdb.set_trace()
        
        f_orb_f.extend(f_orb_f_v) 
        
        #dat = data.iloc(i*ndat/nchunks:(i+1)*ndat/nchunks)
    
    print(len(f_orb_f))
    data['f_orb_f'] = f_orb_f
    

    #Update with values of orbital freq greater than 0.5*10^-4 (in LISA band)
    #selection for LISA frequency band
    data = data.loc[ data["f_orb_f"] > 0.5* 10**(-4) ]
    print('Length of data with selected orbital frequency is: ' + str(len(data)))

    return data

'''POSITIONS'''

'''Han, 2023 triaxial spherical coordinates'''
'''Accreted halo from 6-60 kpc'''

alpha1 = 1.70
alpha2 = 3.09
alpha3 = 4.58
radbreak1 = 11.85
radbreak2 = 29.26
radbreak3 = 100
p = 0.81
q = 0.73

def halo_dens(R_flattened, alpha1, alpha2, alpha3, radbreak1, radbreak2):
        #First break radius

        if 6.0 <= R_flattened < radbreak1:
            p_dis = R_flattened**-alpha1

        #Second break radius
        elif radbreak1 <= R_flattened < radbreak2:
            p_dis = radbreak1**(alpha2-alpha1) * R_flattened**-alpha2
            
        #Third break radius
        elif radbreak2 <= R_flattened <= 60 :
            p_dis = radbreak1**(alpha2-alpha1) * radbreak2**(alpha3-alpha2) * R_flattened**-alpha3
        
        else:
            p_dis = 0
            
        return p_dis * 1/(0.19697150354990667)

def triax_halo(numsamples, p, q, radbreak1, radbreak2, alpha1, alpha2, alpha3):
    l = []
    
    x_list = []
    y_list =[]
    z_list = []
    
    yaw = -24.33 # azimuthal angle, about z-axis
    pitch = -25.39 #tilt, relative to plane
    
    #measured clockwise
        
    while len(l) < numsamples:
        
        # Sample uniformly within a bounding box that encompasses the prolate spheroid
        
        x = np.random.uniform(-60, 60, numsamples)
        y = np.random.uniform(-60, 60, numsamples)
        z = np.random.uniform(-60, 60, numsamples)
        
        point = (x, y, z)
        
        #Rotation matrices for z and y axes
        rotation_z = np.array([[np.cos(yaw), -np.sin(yaw), 0], [np.sin(yaw), np.cos(yaw), 0], [0, 0, 1]])

        rotation_y = np.array([[np.cos(pitch), 0, np.sin(pitch)],[0, 1, 0], [-np.sin(pitch), 0, np.cos(pitch)]])
        
        #Apply rotation
        rotated_point = np.dot(rotation_z, point)
        rotated_point = np.dot(rotation_y, rotated_point)
        
        x,y,z = rotated_point
        
        R_flattened = np.sqrt(x**2 + y**2 / p**2 + z**2 / q**2) # p = b/a ; q = c/a
        
        u = np.random.uniform(0, 1, 100*numsamples)
    
        p_dis = halo_dens(R_flattened, alpha1, alpha2, alpha3, radbreak1, radbreak2)
        
        indselect, = np.where(u <= p_dis)
        x = x[indselect]
        y = y[indselect]
        z = z[indselect]
        
        print(x)
        x_list.extend(x)
        y_list.extend(y)
        z_list.extend(z)
        
    l = list(zip(x_list, y_list, z_list))
        
    return l 


''' 
Returns spherically symmetric coordinates using Ruiter, 2009 halo density profile
'''
def coords_sphere_ruiter(num_samples):
    l = []
    
    x_list = []
    y_list =[]
    z_list = []
    
    a_naught = 3.5
        
    while len(x_list) < num_samples:

        u =  np.random.uniform(0,1,100 * num_samples)
        phi = np.random.uniform(0,2*np.pi, 100 * num_samples)
        u_theta = np.random.uniform(-1,1, 100 * num_samples)
        theta = np.arccos(u_theta,) #theta =  np.random.uniform(0, np.pi)
        uniform_r= np.random.uniform(0, 30, 100*num_samples) #a value, r, from uniform distribution

        prob_dens = (5/(2*a_naught))*(1+uniform_r/a_naught)**(-3.5)
        
        indselect, = np.where(u <= prob_dens)
        u_select = u[indselect]
        phi_select = phi[indselect]
        theta_select = theta[indselect]
        uniform_r_select = uniform_r[indselect]

        
        r_xy = uniform_r_select *np.sin(theta_select)
        x = np.cos(phi_select ) * r_xy
        y = np.sin(phi_select ) * r_xy
        z = uniform_r_select * np.cos(theta_select)
        
        x_list.extend(x)
        y_list.extend(y)
        z_list.extend(z)
        
    l = list(zip(x_list, y_list, z_list))
    
    l = np.array(l)
    
    #import pdb
    
    #pdb.set_trace()
    
    return l[0:num_samples]

''' 
Returns uniform spherical coordinates 
'''
def coords_sphere(cx, cy,cz, num_samples, rmin, rmax, epsilon):
    l = []
    R =  rmax - rmin
    for i in range(num_samples):
        u =  np.random.uniform(-1,1)
        lamba =  np.random.uniform(0,1)
        phi = np.random.uniform(0,2*np.pi)
        x = cx + R*lamba**(1/3)*np.sqrt(1-u**2)*np.cos(phi)
        y = cy + R*lamba**(1/3)*np.sqrt(1-u**2)*np.sin(phi)
        z = cz + R*lamba**(1/3)*u / epsilon**2
        l.append((x,y,z))
            
    return l

'''Disk Distribution, McMillan 2011'''

def disk_pos(numsamples, z_d, R_d, cen_surf_dens):
    
    phi = np.random.uniform(0, 2*np.pi, numsamples)
        
    u_z =  np.random.uniform(0,1,numsamples)
    u_r = np.random.uniform(0,1,numsamples)
        
    z_samp = -z_d * np.log(1-u_z)
    r_samp = -R_d * np.log(1-u_r)
        
    zsign = np.random.uniform(0,1,numsamples)
    z_samp[zsign < 0.5] = -1*z_samp[zsign < 0.5]
    
    x,y,z = (r_samp * np.cos(phi), r_samp *np.sin(phi) ,z_samp)
    
    l = [(x_i, y_i, z_i) for x_i, y_i, z_i in zip(x, y, z)]
    
    return np.array(l)

'''Returns points on the bulge based on McMillan 2011, using rejection sampling'''
alpha = 1.8
r_naught = 0.075
r_cut = 2.1
q = 0.5
p_b_naught = 9.93* 10**10 #scale density
r_vals = np.linspace(0,10,num=500) #range?
z_vals = np.linspace(-5,5,num=500) 


def bulge(numsamples):
    
    l = []
    
    while len(l) < numsamples:
        uniform_r= np.random.uniform(0, 10, numsamples) #a value, r, from uniform distribution
        uniform_z = np.random.uniform(-5, 5,size=numsamples) 


        for r_s, z_s in zip(uniform_r, uniform_z):
            theta = np.pi * np.random.uniform(0,1)
            phi = np.random.uniform(0, 2*np.pi)
            r_prime = np.sqrt(r_s**2+(z_s/q)**2)
            u = np.random.uniform(0,1) #assuming q(r,z) is uniform distribution at 1
            prob_dens = p_b_naught/(1+r_prime/r_naught)**(1.8) * np.exp(-(r_prime/r_cut)**2)
            if prob_dens >= u:
                x,y,z = (r_s * np.sin(theta) * np.cos(phi), r_s * np.sin(theta) *np.sin(phi) ,z_s)
                l.append((x,y,z))
            
    return l[0:numsamples] 



'''CALCULATING STRAIN, SNR, AND CHIRP MASS'''

#First, calculate strain to add to database
def calc_strain(selected_DWDs, component):
    
    n = len(selected_DWDs)
        
    if component == 'bulge':
        pos =  bulge(n)
        setoff_pos = pos - np.array([8,0,0]) #positions relative to the sun
        
    if component == 'in situ halo':
        pos =  coords_sphere_ruiter(n)
        
        setoff_pos = pos - np.array([8,0,0]) #positions relative to the sun
        
    if component == "GSE halo":
        pos = triax_halo(n, p, q, radbreak1, radbreak2, alpha1, alpha2, alpha3)
        setoff_pos = pos - np.array([8,0,0]) #positions relative to the sun
        
    elif component == 'thin disk':
        pos = disk_pos(n, 0.3, 2.6, 816.6)
        setoff_pos = pos - np.array([8,0,0]) #positions relative to the sun
        
    elif component == 'thick disk':
        pos = disk_pos(n, 0.9, 3.6, 209.5)
        setoff_pos = pos - np.array([8,0,0]) #positions relative to the su
    
    #print statement
    m_1 =  selected_DWDs['mass_1'].values *u.Msun
    
    m_2 = selected_DWDs['mass_2'].values * u.Msun

    ecc = selected_DWDs['ecc'].values
    #print(np.sum(utils.peters_g(3,ecc)))
    
    f_orb_f = selected_DWDs['f_orb_f'].values  * u.Hz

    dist = np.linalg.norm(setoff_pos, axis=1) * u.kpc #luminosity distances

    sources = source.Source(m_1=m_1, m_2=m_2, ecc=ecc, dist=dist, f_orb=f_orb_f)
    h_0_n = sources.get_h_0_n(harmonics=2) #unsure about harmonics here
    
    selected_DWDs['strain'] = h_0_n

    return selected_DWDs

#Calculate chirp mass to add to database
def calc_chirp_mass(selected_DWDs):
    
    m_1 =  selected_DWDs['mass_1'].values *u.Msun
    
    m_2 = selected_DWDs['mass_2'].values * u.Msun

    chirp = utils.chirp_mass(m_1, m_2)
    
    selected_DWDs['chirp mass'] = chirp

    return selected_DWDs

'''
Function to assign positions and calaculate SNR using present day orbital frequency
'''
def calc_SNR(selected_DWDs, component):
    
    n = len(selected_DWDs)
    
    if component == 'bulge':
        pos =  bulge(n)
        setoff_pos = pos - np.array([8,0,0]) #positions relative to the sun
        
    if component == 'in situ halo':
        pos =  coords_sphere_ruiter(n)
        setoff_pos = pos - np.array([8,0,0]) #positions relative to the sun
        
    if component == "GSE halo":
        pos = triax_halo(n, p, q, radbreak1, radbreak2, alpha1, alpha2, alpha3)
        setoff_pos = pos - np.array([8,0,0]) #positions relative to the sun
        
    elif component == 'thin disk':
        pos = disk_pos(n, 0.3, 2.6, 816.6)
        setoff_pos = pos - np.array([8,0,0]) #positions relative to the sun
        
    elif component == 'thick disk':
        pos = disk_pos(n, 0.9, 3.6, 209.5)
        setoff_pos = pos - np.array([8,0,0]) #positions relative to the sun
    
    x_pos = setoff_pos[:,0]
    
    y_pos = setoff_pos[:,1]
    
    z_pos = setoff_pos[:,2]
    
    selected_DWDs['x_pos'] = x_pos
    
    selected_DWDs['y_pos'] = y_pos

    selected_DWDs['z_pos'] = z_pos
    
    m_1 =  selected_DWDs['mass_1'].values *u.Msun
    
    m_2 = selected_DWDs['mass_2'].values * u.Msun

    ecc = selected_DWDs['ecc'].values
    
    f_orb_f = selected_DWDs['f_orb_f'].values  * u.Hz

    dist = np.linalg.norm(setoff_pos, axis=1) * u.kpc #luminosity distances

    selected_DWDs['dist'] = dist
    
    sources = source.Source(m_1=m_1, m_2=m_2, ecc=ecc, dist=dist, f_orb=f_orb_f)
    snr = sources.get_snr(verbose=True) 
    
    return snr

'''
Function to select SNRs greater than 7
'''
def select_SNR(selected_DWDs, component):
    #add snr as column,
    snr = calc_SNR(selected_DWDs, component)
    
    selected_DWDs['snr'] = snr.tolist()
   
    #unselectedDWDS = selected_DWDs['snr']

    selected_DWDs_SNR = selected_DWDs.loc[selected_DWDs['snr'] > 7]  #select those greater than 7

    return selected_DWDs, selected_DWDs_SNR #save to file here


def LISA_detectability(filenameofpopulation, M_component, component):
    
    '''Accessing all the data'''
    #Loading file into pandas dataframe and printing
        
    pp  = pd.read_hdf(filenameofpopulation, key='conv', mode='r', errors='strict') 
    
    pp = pp.loc[:,['bin_num', 'tphys', 'mass_1', 'mass_2', 'sep', 'porb', 'ecc']]
    #import pdb
    
    #pdb.set_trace()
    
    #Finding mass of population
    pp_mass_binaries = pd.read_hdf(filenameofpopulation, key="mass_stars",mode='r', errors='strict')
    pp_mass_binaries = max(np.array(pp_mass_binaries))[0]

    #Formation efficiency
    n_pp = len(pp) #number of DWDS
    n_DWD_pp = n_pp/(pp_mass_binaries * u.Msun)

    #Number of DWDs from population
    N_DWD_pp = int(n_DWD_pp * M_component)
    print('The number of double white dwarfs from the '+ ' population is: ' + str(N_DWD_pp)) 
    
    #Binary evolution
    '''Sampling from a population using number of calculated DWDs'''
    pp_s = pp.sample(N_DWD_pp, replace=True)
    
    '''Assigning ages'''
    pp_s_t = findtimes(pp_s, component)
    
    pp_s_t['age'].to_hdf(f"{component.replace(' ', '_')}_age_double_white_dwarfs_empirical.h5", 
                         key= f"{component.replace(' ', '_')}_age", mode='a') 
    pp_s = [] #clear memory
    
    '''Updated data to select for those with appropriate gravitational evolution time 
    and within the LISA band'''
    pp_se = select_DWDs(pp_s_t)
    
    pp_s_t = [] #clear memory
    
    '''Calculating strain'''
    pp_se_st = calc_strain(pp_se, component)
                    
    pp_se = [] #clear memory
                    
    '''Calculating chirp mass'''  
                    
    pp_se_st_m = calc_chirp_mass(pp_se_st)
                    
    pp_se_st = []
                    
    '''Calculating SNR'''
    pp_sel, pp_sel_SNR  = select_SNR(pp_se_st_m, component)
    
    pp_sel = pp_sel.loc[:,['bin_num', 'dist', 'tphys', 'mass_1', 'mass_2',
                           'age', 'f_orb_f', 'snr', 'strain', 'chirp mass', 'x_pos', 'y_pos', 'z_pos']]
    
    pp_sel_SNR = pp_sel_SNR.loc[:,['bin_num', 'dist', 'tphys', 'mass_1', 'mass_2', 'sep', 'porb', 'ecc',
                           'age', 'f_orb_f', 'snr', 'strain', 'chirp mass', 'x_pos', 'y_pos', 'z_pos']]
    
    pp_se_st_m = [] #clear memory
        
    return   N_DWD_pp, pp_sel, pp_sel_SNR 

def results(M_component, component):
    
    DWD_types = ["HeHe", "HeCo", "CoCo", "ONeX"]
    if component in ['in situ halo', 'thick disk']:
        filenames = [hh_filename_10,coh_filename_10,coco_filename_10,oho_filename_10]
    elif component in "GSE halo":
        filenames = [hh_filename_6,coh_filename_6,coco_filename_6,oho_filename_6]
    elif component in ["thin disk", "bulge"]:
        filenames = [hh_filename_14 ,coh_filename_14 ,coco_filename_14 ,oho_filename_14]
    
    N_DWD_list = []
    
    print('The detectability for the ' + f"{component}" + ' is as follows:')

    for DWD, fname in zip(DWD_types, filenames):
        
        print(f"{DWD} white dwarfs in {component}")
        
        N_DWD, DWDdataunsel, DWDdata = LISA_detectability(fname, M_component, component)
        
        DWDdataunsel.to_hdf(f"{component.replace(' ', '_')}_original_double_white_dwarfs_empirical.h5", 
                            key=f"{DWD}_{component.replace(' ', '_')}", mode='a') 
            
        DWDdataunsel = []
        
        DWDdata.to_hdf(f"{component.replace(' ', '_')}_double_white_dwarfs_empirical.h5",
                       key=f"{DWD}_{component.replace(' ', '_')}", mode='a')
        
        print("Length of DWD's with SNR > 7 is " + str(len(DWDdata)), end='\n')
        
        N_DWD_list.append(N_DWD)
        
        DWDdata = []
        
    N_DWD_df = pd.DataFrame(N_DWD_list, columns = ["N_DWD"])
    
    N_DWD_df.to_hdf(f"{component.replace(' ', '_')}_number_double_white_dwarfs_empirical.h5", key = "number_white_dwarfs")
    
'''
def results_alt(filenameofpopulation, M_component, component):
    
    print('The detectability for the ' + f"{component}" + ' is as follows:')
    
    N_DWD_list = []
    
    if fname in [hh_filename_1, hh_filename_14, hh_filename_9]:
        
        print("Helium DWDs" + " in " + f"{component}")
        
        N_DWD_hh, hh_data = LISA_detectability(fname, M_component, component)

        hh_data.to_hdf(
            f"{component.replace(' ', '_')}_double_white_dwarfs.h5" if ' ' in component else f"{component}_double_white_dwarfs.h5", key=f"HeHe_{component.replace(' ', '_')}", 
            mode='a'
        )
        
        N_DWD_list.append(N_DWD_hh)
        
        hh_data = []
            
    elif fname in [coh_filename_1, coh_filename_14, coh_filename_9]:
    
        print("CO and He DWDs" + ' in' + f"{component}")
                  
        N_DWD_coh, coh_data = LISA_detectability(fname, M_component, component)
                  
        coh_data.to_hdf(
            f"{component.replace(' ', '_')}_double_white_dwarfs.h5" if ' ' in component else f"{component}_double_white_dwarfs.h5", key=f"CoHe_{component.replace(' ', '_')}", 
            mode='a'
        )
        
        N_DWD_list.append(N_DWD_coh)
        
        coh_data = [] #wipe out data
            
    elif fname in [coco_filename_1, coco_filename_14, coco_filename_9]:
        print("CO and CO DWDs" + ' in' + f"{component}")
            
        N_DWD, coco_data = LISA_detectability(fname, M_component, component)
        
        #writing data of carbon_oxygen/carbon oxygen white dwarfs w/ SNR>7 to hdf5 file under specified key
        coco_data.to_hdf(
            f"{component.replace(' ', '_')}_double_white_dwarfs.h5" if ' ' in component else f"{component}_double_white_dwarfs.h5", key=f"CoCo_{component.replace(' ', '_')}", 
            mode='a'
        )
        
        N_DWD_list.append(N_DWD_coco)
        
        coco_data = [] #wipe out data
          
    elif filenamefpopulation in [oho_filename_1, oho_filename_14, oho_filename_9]:
        print("One and  DWDs" + ' in' + f"{component}")
        
        N_DWD, oho_data = LISA_detectability(fname, M_component, component)
        num_comp.append(N_DWD_oho_comp)
        
        #writing data of oxygen_neon/helium white dwarfs w/ SNR>7 to hdf5 file under specified key
        
        oho_data.to_hdf(
            f"{component.replace(' ', '_')}_double_white_dwarfs.h5" if ' ' in component else f"{component}_double_white_dwarfs.h5", key=f"ONeX_{component.replace(' ', '_')}", 
            mode='a'
        )
        
        N_DWD_list.append(N_DWD_oho)
        
        oho_data = [] #wipe out data

    N_DWD_df = pd.DataFrame(N_DWD_list, columns = ["N_DWD"])
    
    N_DWD_df.to_hdf(f"{component.replace(' ', '_')}_number_double_white_dwarfs.h5" if ' ' in component else f"{component}_number_double_white_dwarfs.h5", key = "number_white_dwarfs")
'''                      
                       
if __name__ == '__main__':
    
    results(M_GSE_halo, "GSE halo")
    #results(M_in_situ_halo, "in situ halo") #in situ halo, Ruiter
    #results(M_disk_thin, "thin disk")
    #results(M_disk_thick, "thick disk")
    #results(M_bulge, "bulge")
                       