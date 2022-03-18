import numpy as np
import sfs
import matplotlib.pyplot as plt
from data_lib import soundfield_generation as sg
import os
#os.environ['CUDA_VISIBLE_DEVICES']=''
c_complex = 343
pi_complex = np.pi

# Soundfield params (this is not right place)
nfft = 128  # Number of fft points
d = 0.063  # Spacing between sensors
c = 343  # sound speed at 20 degrees
f_s = 1500  # Maximum frequency to be considered in Hz
s_r = 2 * f_s  # Sampling rate
# Frequency axis
f_axis = np.fft.rfftfreq(nfft, 1/s_r)
f_axis = f_axis[2:]
N_freqs = len(f_axis)
wc = 2 * np.pi * f_axis
N = 41

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica"],
    'font.size': 20})

# Circular array parameters ###########################################################################################
N_lspks = 64  # Numbgrider of loudspeakers
grid = sfs.util.xyz_grid([-2, 2], [-2, 2], 0, spacing=0.0201005025125629)
radius = 1  # in m
array = sfs.array.circular(N_lspks, radius)
array_pos = array.x
theta_l = np.zeros(len(array_pos))
for n in range(len(array_pos)):
    _, theta_l[n] = sg.cart2pol(array_pos[n, 0], array_pos[n, 1])

N_sample = 200
x = np.linspace(-2, 2, N_sample)
grid_x, grid_y = np.meshgrid(x, x)
point = np.array([grid_x.ravel(), grid_y.ravel(), np.zeros_like(grid_x.ravel())]).T
N_pts = len(grid_x.ravel())

# Extract points corresponding to interior field w.r.t. the array
first = True
for n_p in range(point.shape[0]):
    r_point, theta_point = sg.cart2pol(point[n_p, 0], point[n_p, 1])
    if r_point < radius:
        if first:
            point_lr = np.expand_dims(point[n_p], axis=0)
            idx_lr = np.expand_dims(n_p, axis=0)
            first =False
        else:
            point_lr = np.concatenate([point_lr, np.expand_dims(point[n_p], axis=0)])
            idx_lr = np.concatenate([idx_lr, np.expand_dims(n_p, axis=0)])

# Extract control points
point_grid = point[np.arange(0,point.shape[0],21),:] # was 41


N_lr_pts = len(point_lr)
x = np.linspace(-1, 1, 20)
x, y = np.meshgrid(x, x)
x = x.ravel()
y = y.ravel()
rho, _ = sg.cart2pol(x, y)
rho_bool = rho < radius
x, y = x[rho_bool], y[rho_bool]
point_cp = np.array([x, y, np.zeros(x.shape)]).transpose()
idx_cp = np.zeros(point_cp.shape[0], dtype=int)
for n_p in range(len(point_cp)):
    idx_cp[n_p] = np.argmin(np.linalg.norm(point_cp[n_p] - point_lr, axis=1))

#print(str(len(x.ravel())), 'control points')

plot_setup = False
if plot_setup:
    plt.figure(figsize=(10, 10))
    plt.plot(point[:, 0], point[:, 1], 'r*')
    plt.plot(point_lr[:, 0], point_lr[:, 1], 'g*')
    plt.plot(point_cp[:, 0], point_cp[:, 1], 'b*')
    plt.plot(array_pos[:, 0], array_pos[:, 1], 'k*')
    plt.show()
print(str(len(point_cp)) + ' control points')
################## Test sources #######################################
# Generate sources
step_radius = 0.1
radius_sources_train = np.arange(1.5, 3.5, step_radius)
radius_sources_test = np.arange(1.5 + (step_radius / 2), 3.5 + step_radius + (step_radius / 2), step_radius)
n_sources_radius = 128
src_pos_train = np.zeros((len(radius_sources_train) * n_sources_radius, 2))
src_pos_test = np.zeros((len(radius_sources_train), n_sources_radius, 2))

angles = np.linspace(0, 2 * np.pi, n_sources_radius)
for n_r in range(len(radius_sources_train)):
    for n_s in range(n_sources_radius):
        src_pos_train[(n_r * n_sources_radius) + n_s] = sg.pol2cart(radius_sources_train[n_r], angles[n_s])
        src_pos_test[n_r, n_s] = sg.pol2cart(radius_sources_test[n_r], angles[n_s])

plot_setup = False
if plot_setup:
    plt.figure(figsize=(10, 10))
    plt.plot(point_lr[:, 0], point_lr[:, 1], 'g*')
    plt.plot(point_cp[:, 0], point_cp[:, 1], 'b*')
    plt.plot(array_pos[:, 0], array_pos[:, 1], 'k*')
    plt.plot(src_pos_train[:,0], src_pos_train[:,1],'c*')
    plt.plot(src_pos_test[:,:,0], src_pos_test[:,:,1],'r*')
    plt.xlabel('$x [m]$'), plt.ylabel('$y [m]$')
    plt.legend(['eval points','control points','loudspeakers','train sources','test sources'])
    plt.show()