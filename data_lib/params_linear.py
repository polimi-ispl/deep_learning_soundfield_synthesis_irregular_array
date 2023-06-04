import numpy as np
import sfs
import matplotlib.pyplot as plt
from data_lib import utils as sg

c_complex = 343
pi_complex = np.pi
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica"],
    'font.size': 20})

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

# Linear array parameters ###########################################################################################
N_lspks = 64
grid = sfs.util.xyz_grid([-2, 2], [-2, 2], 0, spacing=0.0201005025125629)
spacing = 0.0625  # in m
array = sfs.array.linear(N_lspks, spacing,
                         center=[0.5, 0, 0], orientation=[1, 0, 0])
array_pos = array.x
array_x = array_pos[:, 0][0]  # fixed y coordinate
theta_l = np.zeros(len(array_pos))
for n in range(len(array_pos)):
    _, theta_l[n] = sg.cart2pol(array_pos[n, 0], array_pos[n, 1])

N_sample = grid[0].shape[1]
x = np.linspace(-2, 2, N_sample)
grid_x, grid_y = np.meshgrid(x, x)
point = np.array([grid_x.ravel(), grid_y.ravel(), np.zeros_like(grid_x.ravel())]).T
N_pts = len(grid_x.ravel())


# Extract points corresponding to interior field w.r.t. the array
first = True
for n_p in range(point.shape[0]):
    r_point, theta_point = sg.cart2pol(point[n_p, 0], point[n_p, 1])
    if point[n_p, 0] < array_x:
        if first:
            point_lr = np.expand_dims(point[n_p], axis=0)
            idx_lr = np.expand_dims(n_p, axis=0)
            first = False
        else:
            point_lr = np.concatenate([point_lr, np.expand_dims(point[n_p], axis=0)])
            idx_lr = np.concatenate([idx_lr, np.expand_dims(n_p, axis=0)])

# Extract control points
"""
sample_cp = 395#32
idx_cp = np.arange(0,  len(point_lr), sample_cp) # relative to listening area
point_cp = point_lr[idx_cp]
print(len(idx_cp))
"""

x = np.linspace(-2, 2, N_sample//18)  # was 6
grid_x, grid_y = np.meshgrid(x, x)
grid_x, grid_y = grid_x.ravel(), grid_y.ravel()
point_cp_temp = np.array([grid_x, grid_y, np.zeros(shape=grid_x.shape)]).transpose()
first = True
for n_p in range(point_cp_temp.shape[0]):
    if point_cp_temp[n_p, 0] < array_x:
        if first:
            point_cp = np.expand_dims(point_cp_temp[n_p], axis=0)
            idx_cp = np.expand_dims(n_p, axis=0)
            first = False
        else:
            point_cp = np.concatenate([point_cp, np.expand_dims(point_cp_temp[n_p], axis=0)])
            idx_cp = np.concatenate([idx_cp, np.expand_dims(n_p, axis=0)])

for n_p in range(len(point_cp)):
    idx_cp[n_p] = np.argmin(np.linalg.norm(point_cp[n_p] - point_lr, axis=1))
point_cp = point_lr[idx_cp]
print(len(idx_cp))

# Sources positions training
x_min, x_max = 1, 9
y_min, y_max = -2, 2
x = np.linspace(x_min, x_max, 50)
y = np.linspace(y_min, y_max, 50)
[X, Y] = np.meshgrid(x, y)
src_pos_train = np.array([X.ravel(), Y.ravel()])
N_sources = src_pos_train.shape[1]

x_test = np.linspace(x_min + 0.08, x_max + 0.08, 50)
y_test = np.linspace(y_min, y_max, 50)
[X_test, Y_test] = np.meshgrid(x_test, y_test)
src_pos_test = np.array([X_test.ravel(), Y_test.ravel()])
src_pos_test = src_pos_test[:,0:-1:10] # TEMPPPP
n_src_test = src_pos_test.shape[1]

n_src_train = src_pos_train.shape[1]
"""
plot_setup = False
if plot_setup:
    plt.figure(figsize=(20, 10))
    #plt.plot(point[:, 0], point[:, 1], 'r*')
    plt.plot(point_lr[:, 0], point_lr[:, 1], 'g*')
    plt.plot(point_cp[:, 0], point_cp[:, 1], 'b*')
    plt.plot(array_pos[:, 0], array_pos[:, 1], 'k*')
    plt.plot(src_pos_train[0,:], src_pos_train[1,:],'c*')
    plt.plot(src_pos_test[0,:], src_pos_test[1,:],'r*')
    plt.xlabel('$x [m]$'), plt.ylabel('$y [m]$')
    plt.legend(['Eval points', 'Control points', 'Loudspeakers', 'Train sources', 'Test sources'])
    plt.show()
print(str(len(point_cp)) + ' control points')
"""
