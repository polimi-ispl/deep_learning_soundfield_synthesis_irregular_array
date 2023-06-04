import numpy as np
import sfs
import matplotlib.pyplot as plt
from data_lib import soundfield_generation as sg
c_complex = 343
pi_complex = np.pi
grid = sfs.util.xyz_grid([-0.14, 0.14], [-0.14, 0.14], 0, spacing=0.04)
# Soundfield params (this is not right place)
nfft = 48000  # Number of fft points
c = 343  # sound speed at 20 degrees
f_s = 48000  # Sampling rate
# Frequency axis
f_max_analysis = 1500
f_min_analysis = 50
N_freqs = 63
f_axis_mapping = np.linspace(f_min_analysis,f_max_analysis,N_freqs,dtype=int)
f_axis = np.fft.rfftfreq(f_s,1/f_s)[f_axis_mapping]


wc = 2 * np.pi * f_axis
N = 41

# SETUP
N_ldspk_array = 60 # Number of loudspeaker arrays
R = 1.5
radius = R
l = np.arange(1,N_ldspk_array+1) # Ldspk idx

x_ldspk_array = -R*np.sin(((2*l)-1)*(np.pi/N_ldspk_array))
y_ldspk_array = R*np.cos(((2*l)-1)*(np.pi/N_ldspk_array))

# Microphone array
N_mic_array = 64 # Number of loudspeaker arrays
m = np.arange(1,N_mic_array+1)
x_mic_array = -0.14 + (0.04*np.mod(m-1,8))
y_mic_array = 0.14 - (0.04*np.floor((m-1)/8))
mic_array_points = np.array([x_mic_array,y_mic_array,np.zeros_like(y_mic_array)]).T
array_pos_real = np.array([x_ldspk_array, y_ldspk_array, np.zeros_like(x_ldspk_array)]).T
N_mic_axis = 8


idx_missing = np.zeros(N_ldspk_array*2, dtype=int)
for i in range(0, N_ldspk_array * 2 , 2):
    idx_missing[i] = 1
array_wfs_temp = sfs.array.circular(N=N_ldspk_array * 2, R=R)

# We need to roll the array due to mismatch between definition and array in SFS
roll_factor=-30
array_wfs_temp = sfs.array.as_secondary_source_distribution([np.roll(array_wfs_temp.x, roll_factor,axis=0),
                                                        np.roll(array_wfs_temp.n, roll_factor,axis=0),
                                                        np.roll(array_wfs_temp.a, roll_factor,axis=0)])

array_wfs = sfs.array.as_secondary_source_distribution([np.delete(array_wfs_temp.x, idx_missing==1, axis=0),
                                                        np.delete(array_wfs_temp.n, idx_missing==1, axis=0),
                                                        np.delete(array_wfs_temp.a, idx_missing==1, axis=0)])


theta_l = np.zeros(len(array_pos_real))
for n in range(len(array_pos_real)):
    _, theta_l[n] = sg.cart2pol(array_pos_real[n, 0], array_pos_real[n, 1])
N_sample = 8
array = array_wfs
points = mic_array_points
array_pos = array_pos_real

# GENERATE TEST SOURCES
nx, ny = 50, 50
xv, yv = np.meshgrid(np.linspace(-4,4,nx), np.linspace(-4,4,ny), indexing='ij')
src_pos_train = np.array([np.ravel(xv), np.ravel(yv)]).T
# Remove sources inside listening area
idx_remove = []
for i in range(len(src_pos_train)):
    if np.linalg.norm(src_pos_train[i]) < (R+0.25):
        idx_remove.append(i)
src_pos_train = np.delete(src_pos_train,idx_remove,axis=0)
N_sources_train = len(src_pos_train)
print(str(N_sources_train))

# TEST SOURCES
src_pos_test = src_pos_train+0.08

# Control points
idx_cp = np.arange(0,64,4)

N_sources_test = len(src_pos_test)

PLOT_SETUP = False
print('CONTROL POINTS ' +str(len(idx_cp)))
if PLOT_SETUP:
    plt.figure(figsize=(10,10))
    plt.plot(x_mic_array, y_mic_array, 'k*')
    plt.plot(x_mic_array[idx_cp], y_mic_array[idx_cp], 'c*')
    plt.plot(src_pos_train[:, 0], src_pos_train[:, 1], 'b*')
    plt.plot(src_pos_test[:, 0], src_pos_test[:, 1], 'g*')
    plt.plot(x_ldspk_array, y_ldspk_array, 'k*')

    plt.legend(['Mics','Ctrl Pnts'])
    plt.show()

print('done')
