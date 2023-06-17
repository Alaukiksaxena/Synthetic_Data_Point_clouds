import numpy as np

import matplotlib.pylab as plt
#from pyiron.project import Project
import pandas as pd
import pandas
#from pyiron.table.datamining import PyironTable, TableJob
from random import randint
from random import seed
import random
#from numba import jit
import random
import pickle 
#from sklearn.externals import joblib 
#from pyiron.atomistics.structure.atoms import pyiron_to_ase
#from pyiron.atomistics.structure.atoms import ase_to_pyiron
#from dscribe.descriptors import SOAP
from ase import Atoms
from ase.build import bulk
from mpl_toolkits import mplot3d
import random
from ase.spacegroup import crystal
from scipy.spatial.transform import Rotation as R
import math


# define Cu_percentage
def SOL_percentage(basis,ratio_SOL ):
    no_of_SOL_atoms_per=int((basis.get_number_of_atoms()*ratio_SOL)/(1+ratio_SOL))
    return no_of_SOL_atoms_per

def sustitution_function(basis, ratio_SOL,Type):
    
    if Type == "mix":
        ratio_SOL = 0.33
        mid_z = basis.positions[:,-1].max()/2
        fcc_atom_id = np.argwhere(basis.positions[:,-1] >mid_z).flatten()
        no_of_SOL_atoms_per=int((len(fcc_atom_id)*ratio_SOL)/(1+ratio_SOL))
        SOL_pos_random=random.sample(fcc_atom_id.tolist(), no_of_SOL_atoms_per)        
        basis_altered = basis.copy()
        basis_altered.symbols[SOL_pos_random] = "Mg"
    else:
        
        No_SOL_atoms_SZ = SOL_percentage(basis, ratio_SOL)
        ttl_atoms_sim_box= basis.get_number_of_atoms()
        atm_indx_lst = np.arange(0, ttl_atoms_sim_box)

        SOL_pos_random=random.sample(atm_indx_lst.tolist(), No_SOL_atoms_SZ)

        basis_altered = basis.copy()
        """
        for pos in SOL_pos_random:
            basis_altered[pos] = 'Al'
        """
        basis_altered.symbols[SOL_pos_random] = "Cu"#"Mg"
    
    return basis_altered



def structure( Type,size ):
    
    #from ase.spacegroup import crystal
    if  Type == "bcc":
        basis = bulk.create_ase_bulk('Fe', cubic=True).repeat(size)
        
    if Type == "fcc":
        basis = bulk('Al',a= 4.0479, crystalstructure =  "fcc",cubic=True).repeat(size)

    if Type == "l12":
        #print("l12")
        basis = bulk('Al',a= 4.0479, crystalstructure =  "fcc",cubic=True).repeat(1)
        basis.symbols[0] = "Cu" #"Mg"
        basis = basis.repeat(size)
        #basis.euler_rotate(phi=angles[0,0], theta=angles[1,0], psi=angles[2,0], center="COM")


    if Type == "do3":
        a = 5.74
        DO3 = crystal(( "Al",'Fe',"Fe","Fe"),
                               basis=[(0.00000000,  0.00000000,  0.00000000), (0.5, 0,  0)
                                     ,(-0.25,-0.25,-0.25),(0.25,  0.25,  0.25)],
                               spacegroup=225,
                               cellpar=[a, a, a, 90, 90, 90], size=(size, size ,size))
        basis = ase_to_pyiron(DO3)
        
    if Type == "mix":
        
        basis = bulk('Al',a= 4.0479, crystalstructure =  "fcc",cubic=True)
        basis[0] = "Mg"
        basis.set_repeat([20,20,40])
        mid_z = basis.positions[:,-1].max()/2
        fcc_atom_id = np.argwhere(basis.positions[:,-1] >mid_z).flatten()
        chem_indices = basis.get_chemical_indices()
        basis[fcc_atom_id] = "Al"
        
        
    return basis
        

    
def APT_noise(basis, m, sigma_x_y, sigma_z, eff, angles):
    row = basis.get_positions().shape[0]
    noise_x_y = np.random.normal(m , sigma_x_y , [row ,2]) 
    noise_z = np.random.normal(m, sigma_z, [row ,1])
    noise = np.hstack((noise_x_y, noise_z))
    basis.positions = basis.positions +  noise
    
    if eff != 1:
        #print("hello")
        deff = 1.0 - eff
        atms_remove = int(row*deff)
        atms = random.sample( np.arange(0, row).tolist(), atms_remove)
        del basis[atms]
    #basis.euler_rotate(phi=angles[0,0], theta=angles[1,0], psi=angles[2,0], center="COM")
    return basis

def rotate_axis(angles, order = "xyz" ):
    
    r = R.from_rotvec([math.radians(angles[0]),math.radians(angles[1]),math.radians(angles[2])])
    r.as_euler(order, degrees=True)
    
    x = np.array([1, 0, 0])
    y = np.array([0, 1, 0])
    z = np.array([0, 0, 1])
    
    x_rot = r.apply(x)
    y_rot = r.apply(y)
    z_rot = r.apply(z)
    
    if round(np.dot(x_rot, y_rot),4) != 0:
        
        print(f"not orthogonal x and y with dot {np.dot(x_rot, y_rot)}")
        
    if round(np.dot(z_rot, y_rot),4) != 0:

        print(f"not orthogonal z and y with dot {np.dot(z_rot, y_rot)}")
        
    if round(np.dot(z_rot, x_rot),4) != 0:

        print(f"not orthogonal z and x with dot {np.dot(z_rot, x_rot)}")
        
    return np.array([list(x_rot), list(y_rot), list(z_rot)])
    
    
    

def APT_noise_rot(basis, m, sigma_x_y, sigma_z, eff, angles):
    rot_axis = rotate_axis(angles, order = "xyz" )
    row = basis.get_positions().shape[0]
    rot_arr_x = np.array([list(rot_axis[0])]*basis.positions.shape[0])
    rot_arr_y = np.array([list(rot_axis[1])]*basis.positions.shape[0])
    rot_arr_z = np.array([list(rot_axis[2])]*basis.positions.shape[0])  
    
    noise_x = np.random.normal(m , sigma_x_y , [row ,1]) 
    noise_y = np.random.normal(m , sigma_x_y , [row ,1]) 
    noise_z = np.random.normal(m , sigma_z , [row ,1])     


    noise_add_x = rot_arr_x*noise_x
    noise_add_y = rot_arr_y*noise_y
    noise_add_z = rot_arr_z*noise_z

    basis.positions +=noise_add_x
    basis.positions +=noise_add_y
    basis.positions +=noise_add_z

    
    if eff != 1:
        #print("hello")
        deff = 1.0 - eff
        atms_remove = int(row*deff)
        atms = random.sample( np.arange(0, row).tolist(), atms_remove)
        del basis[atms]

    return basis



def APT_structure(Type, size, ratio_SOL,  m, sigma_x_y, sigma_z,eff,angles,rot  ):
    struct =  structure(Type = Type,size = size)
    
    altered_str = sustitution_function(basis = struct, ratio_SOL = ratio_SOL, Type=Type)
    
    altered_str_copy = altered_str.copy()
    if rot:
        APT_str = APT_noise_rot(basis = altered_str_copy, m = m, sigma_x_y = sigma_x_y, sigma_z = sigma_z, eff = eff, angles = angles  )
    else:
        APT_str = APT_noise(basis = altered_str_copy, m = m, sigma_x_y = sigma_x_y, sigma_z = sigma_z, eff = eff, angles = angles  )
     
    return APT_str 


def SOAP_APT_calculation(basis, spec_atomic_no, rcut, nmax, lmax, periodic_bool,sigma_soap ):

    periodic_soap = SOAP(species= spec_atomic_no, rcut=rcut ,nmax=nmax, lmax=lmax,  sigma = sigma_soap,
                         periodic= periodic_bool,sparse=False)
    
    Soap_vector_length=periodic_soap.get_number_of_features()
    py_to_ase_structure=pyiron_to_ase(basis)
    Soap_vectors = periodic_soap.create(py_to_ase_structure)
    
   
    Soap_vector_dic = np.arange(0, Soap_vector_length, 1)
    t_dict = { }
    for i in range(len(Soap_vector_dic)):
        t_dict['Soap_{}'.format(i)] = []
        
        
    
    for j in range(len(Soap_vectors)):
        for i_s, s in enumerate(Soap_vectors[j]):

            t_dict['Soap_{}'.format(i_s)].append(s)
            
    df_SOAP = pandas.DataFrame(t_dict)
    
    
    
    return df_SOAP



def data_bank(pr, Type,size,sigma_x_y, sigma_z, eff, ratio_SOL):
    if Type == "do3":
        ratio_SOL = [0]
    print(ratio_SOL)
    SOAP_lst = []
    for i in sigma_x_y:
        for j in sigma_z:
            for k in eff:
                for l in ratio_SOL:
                    APT_struct = APT_structure(pr = pr,Type = Type, 
                                                   size = size, ratio_SOL = l,  m=0, sigma_x_y = i, sigma_z=j,eff = k )
                    SOAP = SOAP_APT_calculation(basis = APT_struct , spec_atomic_no = [13,26], rcut = 10, 
                                nmax = 5, lmax = 1 , periodic_bool = True, sigma_soap = 1 )
                    
                    Chem_sym = APT_struct.get_chemical_symbols()
                    SOAP.insert(0, "Spec", Chem_sym, False)

                    SOAP_lst.append(SOAP)
    SOAP_df = pd.concat(SOAP_lst)
    SOAP_df.reset_index(drop=True, inplace=True)
    return SOAP_df