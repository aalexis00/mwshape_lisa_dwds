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

parser = argparse.ArgumentParser()

parser.usage = 'Use this to run the scripts for each component'

parser.add_argument("component", type = str, help= " Returns full dataset of DWD's in component with SNR > 7", choices = ['halo', 'bulge', 'thin disk', 'thick disk'])


args = parser.parse_args()


'''File names for halo'''

#lowest metallicity bin
hh_filename_1 = 'fiducial_10_10_1.h5' #He + He
coh_filename_1 = 'fiducial_11_10_1.h5' # CO + He
coco_filename_1 = 'fiducial_11_11_1.h5' # CO + CO
oho_filename_1 = 'fiducial_12_10_12_1.h5' #ONe + X


'''File names for bulge and thin disk'''

#metallicity bin 14, corresponding to 0.02
hh_filename_14 = 'fiducial_10_10_14.h5'
coh_filename_14 = 'fiducial_11_10_14.h5'
coco_filename_14 = 'fiducial_11_11_14.h5'
oho_filename_14 = 'fiducial_12_10_12_14.h5'

'''File names for thick disk'''

#metallicity bin 9, corresponding to 0.003
hh_filename_9 = 'fiducial_10_10_9.h5'
coh_filename_9 = 'fiducial_11_10_9.h5'
coco_filename_9 = 'fiducial_11_11_9.h5'
oho_filename_9 = 'fiducial_12_10_12_9.h5'

'''Halo information'''

M_halo = 2.6 * 10**8 * u.Msun #from Robin, 2003

agestellarhalo = 14 * 1e9 * u.yr #all formed in one burst according to Robin, 2003

'''Thin disk information''' #McMillan, 2011

cen_surf_dens_thin = 816.6
R_d_thin = 2900
M_disk_thin = 2*np.pi * cen_surf_dens_thin * R_d_thin**2 *u.Msun 

age_thick_disk = 11 * 1e9 * u.yr #all formed in one burst according to Robin, 2003
M_disk_thin

'''Thick disk information''' #McMillan, 2011

cen_surf_dens_thick = 209.5
R_d_thick = 3310
M_disk_thick = 2*np.pi * cen_surf_dens_thick * R_d_thick**2 *u.Msun
M_disk_thick

'''Bulge information''' #McMillan, 2011

M_bulge = 8.9 * 10**9 * u.Msun #McMillan, 2011

age_bulge = 10 * 1e10 * u.yr #all formed in one burst according to Robin, 2003

'''Evolution times'''

#Calculates all gravitational evolution times given the time elapsed since enterred ZAMS
def t_grav_evol(component, timeelapsed, n): 
    if component == 'thin disk': #Thin disk binaries did not form in one burst
        birth = np.random.uniform(0, 10, n) * 1e9 * u.yr
        tgrav = birth - timeelapsed*u.Myr
    else:
        if component == 'thick disk':
            age_component =  11 * 1e9 * u.yr
        if component == 'bulge':
            age_component = 10 * 1e9 * u.yr 
        if component == 'halo':
            age_component =  14 * 1e9 * u.yr
        birth = np.random.uniform(age_component.value - 1e9 , age_component.value,n)
        tgrav = birth* u.yr - timeelapsed*u.Myr
        
    return tgrav.value, birth

#Calculates all gravitational evolution times for given fiducial model
def findtimes(d, component):
    #t = []
    #a = []
    #for i in d['tphys'].values:
    tg, age = t_grav_evol(component, d['tphys'].values, len(d))
    #t.append(tg) #get both thin disk age 
    #a.append(age)
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
    #ndat = (len(m_1))
    
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
        
        '''
        m_1[i*ndat/nchunks:(i+1)*ndat/nchunks]
        m_2[i*ndat/nchunks:(i+1)*ndat/nchunks]
        porb[i*ndat/nchunks:(i+1)*ndat/nchunks]
        f_orb_i[i*ndat/nchunks:(i+1)*ndat/nchunks]
        '''
        
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

'''Positions'''

#Uniformly spaced spherical coordinates 
#helpful link: https://math.stackexchange.com/questions/87230/picking-random-points-in-the-volume-of-sphere-with-uniform-probability/87238#87238

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


def disk_pos(numsamples, z_d, R_d, cen_surf_dens):
    
    l = []
    
    for i in range(numsamples):
        
        theta = np.pi * np.random.uniform(0,1)
        phi = np.random.uniform(0, 2*np.pi)
        
        u_z =  np.random.uniform(0,1)
        u_r = np.random.uniform(0,1)
        
        z_samp = -z_d * np.log(1-u_z)
        r_samp = -R_d * np.log(1-u_r)
        
        zsign = np.random.uniform(0,1)
        if zsign < 0.5:
            z_samp = -1*z_samp
    
        x,y,z = (r_samp * np.sin(theta) * np.cos(phi), r_samp * np.sin(theta) *np.sin(phi) ,z_samp)
        
        l.append((x,y,z))
    
    print(len(l))

    return l 

'''
disk_c = disk_pos(3500, 0.9, 3.6, 209.5) 
disk_coord = np.array(disk_c)
x = disk_coord[:,0]# problem here
y = disk_coord[:,1]
z = disk_coord[:,2]

fig = plt.figure()
ax = fig.add_subplot(1,1,1, projection='3d')
ax.scatter(x,y,z, color='y')
ax.xaxis.set_tick_params(labelsize=12)
ax.yaxis.set_tick_params(labelsize=12)
ax.zaxis.set_tick_params(labelsize=12)

ax.set_xlim(-20,20)
ax.set_ylim(-20,20)
ax.set_zlim(-20,20)

ax.set_xlabel('kpc', fontsize = 14)
ax.set_ylabel('kpc', fontsize = 14)
ax.set_zlabel('kpc', fontsize = 14)
ax.set_box_aspect(aspect=None, zoom=0.8)
'''

''' 
Returns points on the bulge based on McMillan 2011, using rejection sampling

'''


alpha = 1.8
r_naught = 0.075
r_cut = 2.1
q = 0.5
p_b_naught = 9.93* 10**10 #scale density
r_vals = np.linspace(0,10,num=500) #range?
z_vals = np.linspace(-5,5,num=500) 


def bulge_pos(numsamples):
    
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


##Calculating SNR

#First, calculate strain to add to database
def calc_strain(selected_DWDs):
    
    n = len(selected_DWDs)
   
    pos = disk_pos(n,0.3, 2.6, 816.6)
                        
    setoff_pos = pos - np.array([8,0,0]) #positions relative to the sun
    
    m_1 =  selected_DWDs['mass_1'].values *u.Msun
    
    m_2 = selected_DWDs['mass_2'].values * u.Msun

    ecc = selected_DWDs['ecc'].values
    #print(np.sum(utils.peters_g(3,ecc)))
    
    f_orb_f = selected_DWDs['f_orb_f'].values  * u.Hz

    dist = np.linalg.norm(setoff_pos, axis=1) * u.kpc #luminosity distances

    sources = source.Source(m_1=m_1, m_2=m_2, ecc=ecc, dist=dist, f_orb=f_orb_f)
    h_0_n = sources.get_h_0_n(harmonics=[2]) #unsure about harmonics here
    
    selected_DWDs['strain'] = h_0_n

    return selected_DWDs


'''
Function to assign positions and calaculate SNR using present day orbital frequency
'''
def calc_SNR(selected_DWDs, component):
    
    n = len(selected_DWDs)
    
    if component == 'halo':
        pos = coords_sphere(0, 0, 0, n, 10, 30, 0.76)
        setoff_pos = pos - np.array([8,0,0]) #positions relative to the sun
        #print(pos)
        
    elif component == 'thin disk':
        pos = disk_pos(n, 0.3, 2.6, 816.6)
        setoff_pos = pos - np.array([8,0,0]) #positions relative to the sun
        #print(pos)
        
    elif component == 'thick disk':
        pos = disk_pos(n, 0.9, 3.6, 209.5)
        setoff_pos = pos - np.array([8,0,0]) #positions relative to the sun
        #print(pos)
    
    elif component == 'bulge':
        pos = bulge_pos(n)
        setoff_pos = pos - np.array([8,0,0]) #positions relative to the sun
        #print(pos)

    
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

    selected_DWDs = selected_DWDs.loc[selected_DWDs['snr'] > 7]  #select those greater than 7

    return selected_DWDs #save to file here

#find dist, at which detectable

def LISA_dectability(filenameofpopulation, M_component, component):
    
    '''Accessing all the data'''
    #Loading file into pandas dataframe and printing
    pp  = pd.read_hdf(filenameofpopulation, key="conv",mode='r', errors='strict')
    
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
    print('The number of double white dwarfs from the '+ ' population is: ' + str(N_DWD_pp)) #add name of type of binary

    #Binary evolution
    '''Sampling from a population using number of calculated DWDs'''
    pp_s = pp.sample(N_DWD_pp, replace=True)
    
    '''Assigning ages'''
    pp_s_t = findtimes(pp_s, component)
    
    pp_s_t['age'].to_hdf(f"{component}.replace(' ', '_')_age_double_white_dwarfs.h5" if ' ' in filename else filename, key= f"{component.replace(' ', '_')}" if ' ' in filename else filename, mode='w') 
    
    pp_s = [] #clear memory
    
    '''Updated data to select for those with appropriate gravitational evolution time and within the LISA band'''
    pp_se = select_DWDs(pp_s_t)
    
    pp_s_t = []#clear memory
    
    '''Calculating strain'''
    #pp_se_st = calc_strain(pp_se)
    
    '''Calculating SNR'''
    pp_sel = select_SNR(pp_se,component)
     
    pp_sel = pp_sel.loc[:,['bin_num', 'dist', 'tphys', 'mass_1', 'mass_2', 'sep', 'porb', 'ecc', 'age', 'f_orb_f', 'snr']]

    pp_se = [] #clear memory
        
    return   N_DWD_pp, pp_sel


if args.component == 'halo':
    
    print('The detectability for the halo is as follows:')
    
    '''Halo detectability'''

    #setting up dictionaries and values for relevant dataframes
    
    num_halo_df = {} # dataframe for total number of bulge DWDs
    num_halo = []
    num_halo_SNR_df = {}  # dataframe for total number of bulge DWDs w/ SNR > 7
    num_halo_SNR = []
    
    '''Helium DWDs in Halo'''
    print('Helium DWDs in Halo')
    N_DWD_hh_halo, hh_halo = LISA_dectability(hh_filename_1, M_halo, 'halo')
    num_halo.append(N_DWD_hh_halo)
    
    #writing data of helium double white dwarfs w/ SNR>7 to hdf5 file under specified key
    hh_halo.to_hdf('halo_double_white_dwarfs.h5', key='helium_DWDs_halo', mode='a') 
    
    hh_halo = [] #wipe out data

    '''CO + He DWD's in Halo'''
    print('CO + He DWDs in Halo')
    N_DWD_coh_halo, coh_halo = LISA_dectability(coh_filename_1, M_halo, 'halo')
    num_halo.append(N_DWD_coh_halo)

    #writing data of CO+He double white dwarfs w/ SNR>7 to hdf5 file under specified key
    coh_halo.to_hdf('halo_double_white_dwarfs.h5', key='carbon_oxygen/helium_DWDs_halo', mode='a')
    
    coh_halo = [] #wipe out data

    '''CO + CO DWD's in Halo'''
    print('CO + CO DWDs in Halo')
    N_DWD_coco_halo, coco_halo = LISA_dectability(coco_filename_1, M_halo, 'halo')
    num_halo.append(N_DWD_coco_halo)
    
    #writing data of CO+CO double white dwarfs w/ SNR>7 to hdf5 file under specified key
    coco_halo.to_hdf('halo_double_white_dwarfs.h5', key='carbon_oxygen/carbon_oxygen_DWDs_halo', mode='a') 
    
    coco_halo = [] #wipe out data

    '''ONe + He DWD's in Halo'''
    print('ONe + He DWDs in Halo')
    N_DWD_oho_halo, oho_halo = LISA_dectability(oho_filename_1, M_halo, 'halo')
    num_halo.append(N_DWD_oho_halo)

    #writing data of ONe+He double white dwarfs w/ SNR>7 to hdf5 file under specified key
    oho_halo.to_hdf('halo_double_white_dwarfs.h5', key='oxygen/neon_helium_DWDs_halo', mode='a')
    
    oho_halo = [] #wipe out data

    #create databases for total number of halo DWDs as well as number of those w/ SNR>&
    num_halo_df['num_halo'] = num_halo
        
    num_halo_df = pd.DataFrame.from_dict(num_halo_df)
    
    num_halo_df.to_hdf('halo_number_double_white_dwarfs.h5', key='number_halo')
    

    
if args.component == 'bulge':
    
    print('The detectability for the bulge is as follows:')

    '''Bulge detectability'''
    
    #setting up dictionaries and values for relevant dataframes
    
    num_bulge_df = {} # dataframe for total number of bulge DWDs
    num_bulge = []

    '''Helium DWDs in Bulge'''
    print('Helium DWDs in Bulge')
    N_DWD_hh_bulge, hh_bulge = LISA_dectability(hh_filename_14, M_bulge, 'bulge')

    #writing data of helium double white dwarfs w/ SNR>7 to hdf5 file under specified key
    hh_bulge.to_hdf('bulge_double_white_dwarfs.h5', key='helium_DWDs_bulge', mode='a')
    
    num_bulge.append(N_DWD_hh_bulge)
    
    hh_bulge = [] #wipe out data
    
    '''CO + He DWD's in Bulge'''
    print('CO + He DWDs in Bulge')
    N_DWD_coh_bulge, coh_bulge = LISA_dectability(coh_filename_14, M_bulge, 'bulge')
    
    #writing data of carbon_oxygen/helium double white dwarfs w/ SNR>7 to hdf5 file under specified key
    coh_bulge.to_hdf('bulge_double_white_dwarfs.h5', key='carbon_oxygen/helium_DWDs_bulge', mode='a')
    
    num_bulge.append(N_DWD_coh_bulge)
    
    coh_bulge = [] #wipe out data
    
    '''CO + CO DWD's in Bulge'''
    print('CO + CO DWDs in Bulge')
    N_DWD_coco_bulge, coco_bulge = LISA_dectability(coco_filename_14, M_bulge, 'bulge')

    #writing data of carbon_oxygen/carbon_oxygen double white dwarfs w/ SNR>7 to hdf5 file under specified key
    coco_bulge.to_hdf('bulge_double_white_dwarfs.h5', key='carbon_oxygen/carbon_oxygen_DWDs_bulge', mode='a')
    num_bulge.append(N_DWD_coco_bulge)
    
    coco_bulge = []
    
    '''ONe + He DWD's in Bulge'''
    print('ONe + He DWDs in Bulge')
    N_DWD_oho_bulge, oho_bulge = LISA_dectability(oho_filename_14, M_bulge, 'bulge')

    #writing data of oxygen_neon/helium double white dwarfs w/ SNR>7 to hdf5 file under specified key
    oho_bulge.to_hdf('bulge_double_white_dwarfs.h5', key='oxygen_neon/helium_DWDs_bulge', mode='a')
    
    num_bulge.append(N_DWD_oho_bulge)
    

    oho_bulge = [] #wipe out data
    
    #create databases for total number of bulge DWDs as well as number of those w/ SNR>&
    num_bulge_df['num_bulge'] = num_bulge
    
    num_bulge_df = pd.DataFrame.from_dict(num_bulge_df)
    
    num_bulge_df.to_hdf('bulge_number_double_white_dwarfs.h5', key='number_bulge')

    
if args.component == 'thin disk':

    print('The detectability for the thin disk is as follows:')
    
    '''Thin disk detectability'''
    
    #setting up dictionaries and values for relevant dataframes
    num_thin_disk = []
    num_thin_disk_df = {}
    
    '''Helium DWDs in Thin Disk'''
    print('Helium DWDs in Thin Disk')
    N_DWD_hh_thin_disk, hh_thin_disk = LISA_dectability(hh_filename_14, M_disk_thin, 'thin disk')
    
    #writing data of helium double white dwarfs w/ SNR>7 to hdf5 file under specified key
    hh_thin_disk.to_hdf('thin_disk_double_white_dwarfs.h5', key='helium_DWDs_thin_disk', mode='a')
    
    num_thin_disk.append(N_DWD_hh_thin_disk)
    
    hh_thin_disk= [] #wipe out data
    
    '''CO + He DWDs in Thin Disk'''
    print('CO + He DWDs in Thin Disk')
    N_DWD_coh_thin_disk, coh_thin_disk = LISA_dectability(coh_filename_14, M_disk_thin, 'thin disk')
    
    #writing data of CO+He double white dwarfs w/ SNR>7 to hdf5 file under specified key
    coh_thin_disk.to_hdf('thin_disk_double_white_dwarfs.h5', key='carbon_oxygen/helium_DWDs_thin_disk', mode='a')
    
    num_thin_disk.append(N_DWD_coh_thin_disk)
    
    coh_thin_disk = [] #wipe out data

    '''CO + CO DWD's in Thin Disk'''
    print('CO + CO DWDs in Thin Disk')
    N_DWD_coco_thin_disk, coco_thin_disk = LISA_dectability(coco_filename_14, M_disk_thin, 'thin disk')

    #writing data of CO+CO double white dwarfs w/ SNR>7 to hdf5 file under specified key
    coco_thin_disk.to_hdf('thin_disk_double_white_dwarfs.h5', key='carbon_oxygen/carbon-oxygen_DWDs_thin_disk', mode='a')
    
    num_thin_disk.append(N_DWD_coco_thin_disk)
    
    coco_thin_disk = [] #wipe out data
    
    ''''ONe + CO DWD's in Thin Disk'''
    print('ONe + CO DWDs in Thin Disk')
    N_DWD_oho_thin_disk, oho_thin_disk = LISA_dectability(oho_filename_14, M_disk_thin, 'thin disk')

    #writing data of ONe+ He double white dwarfs w/ SNR>7 to hdf5 file under specified key
    oho_thin_disk.to_hdf('thin_disk_double_white_dwarfs.h5', key='oxygen_neon/helium_DWDs_thin_disk', mode='a')
    
    num_thin_disk.append(N_DWD_oho_thin_disk)

    oho_thin_disk = [] #wipe out data
    
    #create databases for total number of thin disk DWDs as well as number of those w/ SNR>7
    num_thin_disk_df['num_thin_disk'] = num_thin_disk
    
    num_thin_disk_df = pd.DataFrame.from_dict(num_thin_disk_df)
    
    num_thin_disk_df.to_hdf('thin_disk_number_double_white_dwarfs.h5', key='number_thin_disk')

    
if args.component == 'thick disk':

    print('The detectability for the thick disk is as follows:')

    '''Thick disk detectability'''

    num_thick_disk = []
    num_thick_disk_df = {}
    
    '''Helium DWDs in Thick Disk'''
    print('Helium DWDs Thick Disk')
    N_DWD_hh_thick_disk, hh_thick_disk = LISA_dectability(hh_filename_9, M_disk_thick, 'thick disk')

    #writing data of helium double white dwarfs w/ SNR>7 to hdf5 file under specified key
    hh_thick_disk.to_hdf('thick_disk_double_white_dwarfs.h5', key='helium_DWDs_thick_disk', mode='a')
    
    num_thick_disk.append(N_DWD_hh_thick_disk) #total number of helium DWDs in thick disk

    hh_thick_disk = [] #wipe out data
    
    '''CO+He DWDs in Thick Disk'''
    print('CO+He DWDs in Thick Disk')
    N_DWD_coh_thick_disk, coh_thick_disk = LISA_dectability(coh_filename_9, M_disk_thick, 'thick disk')

    #writing data of CO+He double white dwarfs w/ SNR>7 to hdf5 file under specified key
    coh_thick_disk.to_hdf('thick_disk_double_white_dwarfs.h5', key='carbon/oxygen_helium_DWDs_thick_disk', mode='a')
    
    num_thick_disk.append(N_DWD_coh_thick_disk) #total number of CO+He DWDs in thick disk

    coh_thick_disk = [] #wipe out data
    
    '''CO+CO DWDs in Thick Disk'''
    print('CO+CO DWDs in Thick Disk')
    N_DWD_coco_thick_disk, coco_thick_disk = LISA_dectability(coco_filename_9, M_disk_thick, 'thick disk')
    
    #writing data of CO+CO double white dwarfs w/ SNR>7 to hdf5 file under specified key
    coco_thick_disk.to_hdf('thick_disk_double_white_dwarfs.h5', key='carbon/oxygen_carbon/oxygen_DWDs_thick_disk', mode='a')
    
    num_thick_disk.append(N_DWD_coco_thick_disk) #total number of CO+CO DWDs in thick disk
    
    coco_thick_disk = [] #wipe out data
    
    '''ONe+He DWDs in Thick Disk'''
    print('ONe+He DWDs in Thick Disk')
    N_DWD_oho_thick_disk, oho_thick_disk = LISA_dectability(oho_filename_9, M_disk_thick, 'thick disk')

    oho_thick_disk.to_hdf('thick_disk_double_white_dwarfs.h5', key='oxygen_neon/helium_DWDs_thick_disk', mode='a')
    num_thick_disk.append(N_DWD_oho_thick_disk)
    
    oho_thick_disk = [] #wipe out data
    
    #create databases for total number of thick disk DWDs as well as number of those w/ SNR>&
    num_thick_disk_df['num_thick_disk'] = num_thick_disk
    
    num_thick_disk_df = pd.DataFrame.from_dict(num_thick_disk_df)
    
    num_thick_disk_df.to_hdf('thick_disk_number_double_white_dwarfs.h5', key='number_thick_disk')


    
'''
writefile = f"{args.component}_double_white_dwarfs.h5"
numberwrite = f"{args.component}number_double_white_dwarfs.h5"
'''