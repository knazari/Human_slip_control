import glob
import time
import numpy as np
import pandas as pd
from scipy import integrate
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
from dtw import *

def smooth(x,window_len=11,window='hanning'):
    s = np.r_[x[window_len-1:0:-1],x,x[-2:-window_len-1:-1]]
    if window == 'flat': #moving average
        w=np.ones(window_len,'d')
    else:
        w=eval('np.'+window+'(window_len)')
    y=np.convolve(w/w.sum(),s,mode='valid')
    
    return y[int((window_len/2-1)):-int((window_len/2))]


user_path = "/home/kia/catkin_ws/src/data_collection_human_test/data/subject_002/baseline"
files = sorted(glob.glob(user_path + "/*"))

rotation_full     = []
max_normal_force  = []
acceleration_full = []

for file in files:
    meta_data       = pd.read_csv(file + '/meta_data.csv')
    hand_imu        = pd.read_csv(file + '/hand_imu.csv')
    object_imu      = pd.read_csv(file + '/object_imu.csv')
    xela_data       = pd.read_csv(file + '/xela.csv')[['txl1_z', 'txl2_z', 'txl3_z', 'txl4_z', 'txl5_z', 'txl6_z', 'txl7_z', 'txl8_z', 'txl9_z', 'txl10_z',
                                                        'txl11_z', 'txl12_z', 'txl13_z', 'txl14_z', 'txl15_z', 'txl16_z']]
    
    start_index = meta_data['start_index'][0]
    end_index   = meta_data['end_index'][0]

    # Object rotation
    object_imu_rot  = R.from_quat(object_imu[['q1', 'q2', 'q3', 'q0']]).as_euler('zyx', degrees=True)
    object_rotation = smooth(object_imu_rot[:, 0] - object_imu_rot[0, 0])
    rotation_full.append(object_rotation - object_rotation[0])

    # Hand acceleration
    a_y_smoothed = -smooth(np.array(hand_imu['acc_y']))
    v_y = integrate.cumtrapz(a_y_smoothed, np.arange(len(a_y_smoothed))*0.017, initial=0)
    acceleration = a_y_smoothed - a_y_smoothed[0]
    acceleration_full.append(acceleration)
    
    # Max grip force
    xela_data = np.array(xela_data[start_index:end_index])
    max_normal_force.append(np.max(np.sum(xela_data, axis=1)))

    fig, ax = plt.subplots(2)
    ax[0].plot(object_rotation)
    ax[1].plot(acceleration)

    ax[0].vlines(start_index, -max(abs(object_rotation)), max(abs(object_rotation)), color='c')
    ax[0].vlines(end_index, -max(abs(object_rotation)), max(abs(object_rotation)), color='c')
    ax[1].vlines(start_index, -max(abs(acceleration)), max(abs(acceleration)), color='c')
    ax[1].vlines(end_index, -max(abs(acceleration)), max(abs(acceleration)), color='c')

plt.show()