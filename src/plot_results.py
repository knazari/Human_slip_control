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


user_path = "/home/kia/catkin_ws/src/data_collection_human_test/data/pilot_test2/subject_001/baseline"
files = sorted(glob.glob(user_path + "/*"))

rotation_full     = []
max_normal_force  = []
acceleration_full = []
fig, ax = plt.subplots(2)

for index, file in enumerate(files):
    if index == 0:
        # start_index = 250
        # end_index   = 300
        continue
    elif index ==1:
        # start_index = 278
        # end_index   = 342
        continue
    elif index ==2:
        start_index = 480
        end_index   = 580
    elif index ==3:
        start_index = 480
        end_index   = 580
    elif index ==4:
        start_index = 430
        end_index   = 530
    elif index ==5:
        start_index = 350
        end_index   = 450
    elif index ==6:
        start_index = 350
        end_index   = 450
    elif index ==7:
        start_index = 360
        end_index   = 460
    elif index ==8:
        start_index = 375
        end_index   = 475
    elif index ==9:
        start_index = 360
        end_index   = 460
    
    meta_data       = pd.read_csv(file + '/meta_data.csv')
    hand_imu        = pd.read_csv(file + '/hand_imu.csv')
    object_imu      = pd.read_csv(file + '/object_imu.csv')
    xela_data       = pd.read_csv(file + '/xela.csv')[['txl1_z', 'txl2_z', 'txl3_z', 'txl4_z', 'txl5_z', 'txl6_z', 'txl7_z', 'txl8_z', 'txl9_z', 'txl10_z',
                                                        'txl11_z', 'txl12_z', 'txl13_z', 'txl14_z', 'txl15_z', 'txl16_z']]
    
    # start_index = meta_data['start_index'][0]
    # end_index   = meta_data['end_index'][0]

    # Object rotation
    object_imu_rot  = R.from_quat(object_imu[['q1', 'q2', 'q3', 'q0']]).as_euler('zyx', degrees=True)
    object_rotation = smooth(object_imu_rot[start_index:end_index, 0] - object_imu_rot[start_index, 0])
    rotation_full.append(object_rotation - object_rotation[0])

    # Hand acceleration
    a_y_smoothed = -smooth(np.array(hand_imu['acc_y'])[start_index:end_index])
    v_y = integrate.cumtrapz(a_y_smoothed, np.arange(len(a_y_smoothed))*0.017, initial=0)
    acceleration = a_y_smoothed - a_y_smoothed[0]
    acceleration_full.append(acceleration)
    
    # Max grip force
    xela_data = np.array(xela_data[start_index:end_index])
    max_normal_force.append(np.max(np.sum(xela_data, axis=1)))

    
    # fig, ax = plt.subplots(2)
    # ax[0].plot(object_rotation, c='b')
    # ax[1].plot(acceleration, c='b')

    # ax[0].vlines(start_index, -max(abs(object_rotation)), max(abs(object_rotation)), color='c')
    # ax[0].vlines(end_index, -max(abs(object_rotation)), max(abs(object_rotation)), color='c')
    # ax[1].vlines(start_index, -max(abs(acceleration)), max(abs(acceleration)), color='c')
    # ax[1].vlines(end_index, -max(abs(acceleration)), max(abs(acceleration)), color='c')

    # ax[0].set_xticks(np.arange(0, len(object_rotation), 50))
    # ax[1].set_xticks(np.arange(0, len(object_rotation), 50))

rotation_full_aligned = []
acceleration_full_aligned = []

for i in range(len(rotation_full)):
    alignment = dtw(rotation_full[1],rotation_full[i], keep_internals=True, 
    step_pattern=rabinerJuangStepPattern(6, "c"))

    rebuilt_vec = np.zeros(len(alignment.index2))
    for index, val in enumerate(alignment.index2):
        rebuilt_vec[index] = rotation_full[i][val]
    
    rotation_full_aligned.append(rebuilt_vec)

for i in range(len(acceleration_full)):
    alignment = dtw(acceleration_full[3],acceleration_full[i], keep_internals=True, 
    step_pattern=rabinerJuangStepPattern(6, "c"))

    rebuilt_vec = np.zeros(len(alignment.index2))
    for index, val in enumerate(alignment.index2):
        rebuilt_vec[index] = acceleration_full[i][val]
    
    acceleration_full_aligned.append(rebuilt_vec)

acceleration_full_aligned = np.array(acceleration_full_aligned)
rotation_full_aligned = np.array(rotation_full_aligned)
for i in range(len(acceleration_full_aligned)):
    ax[0].plot(smooth(rotation_full_aligned[i]), c='b')
    ax[1].plot(smooth(acceleration_full_aligned[i]), c='b')

user_path = "/home/kia/catkin_ws/src/data_collection_human_test/data/pilot_test2/subject_001/controlled"
files = sorted(glob.glob(user_path + "/*"))

rotation_full     = []
max_normal_force  = []
acceleration_full = []

for index, file in enumerate(files):
    if index == 0:
        start_index = 380
        end_index   = -25
    elif index ==1:
        start_index = 425
        end_index   = -30
    elif index ==2:
        start_index = 400
        end_index   = -10
    elif index ==3:
        start_index = 455
        end_index   = -5
    elif index ==4:
        start_index = 400
        end_index   = -1
    elif index ==5:
        start_index = 580
        end_index   = -10
    elif index ==6:
        start_index = 380
        end_index   = 550
    elif index ==7:
        start_index = 420
        end_index   = -1
    elif index ==8:
        start_index = 430
        end_index   = -1
    elif index ==9:
        start_index = 460
        end_index   = -1
    
    meta_data       = pd.read_csv(file + '/meta_data.csv')
    hand_imu        = pd.read_csv(file + '/hand_imu.csv')
    object_imu      = pd.read_csv(file + '/object_imu.csv')
    xela_data       = pd.read_csv(file + '/xela.csv')[['txl1_z', 'txl2_z', 'txl3_z', 'txl4_z', 'txl5_z', 'txl6_z', 'txl7_z', 'txl8_z', 'txl9_z', 'txl10_z',
                                                        'txl11_z', 'txl12_z', 'txl13_z', 'txl14_z', 'txl15_z', 'txl16_z']]
    
    # start_index = meta_data['start_index'][0]
    # end_index   = meta_data['end_index'][0]

    # Object rotation
    object_imu_rot  = R.from_quat(object_imu[['q1', 'q2', 'q3', 'q0']]).as_euler('zyx', degrees=True)
    object_rotation = smooth(object_imu_rot[start_index:end_index, 0] - object_imu_rot[start_index, 0])
    rotation_full.append(object_rotation - object_rotation[0])

    # Hand acceleration
    a_y_smoothed = -smooth(np.array(hand_imu['acc_y'])[start_index:end_index])
    v_y = integrate.cumtrapz(a_y_smoothed, np.arange(len(a_y_smoothed))*0.017, initial=0)
    acceleration = a_y_smoothed - a_y_smoothed[0]
    acceleration_full.append(acceleration)
    
    # Max grip force
    xela_data = np.array(xela_data[start_index:end_index])
    max_normal_force.append(np.max(np.sum(xela_data, axis=1)))

    # ax[0].plot(object_rotation, c='r')
    # ax[1].plot(acceleration, c='r')

    # ax[0].set_xticks(np.arange(0, len(object_rotation), 50))
    # ax[1].set_xticks(np.arange(0, len(object_rotation), 50))

rotation_full_aligned = []
acceleration_full_aligned = []

for i in range(len(rotation_full)):
    alignment = dtw(rotation_full[1],rotation_full[i], keep_internals=True, 
    step_pattern=rabinerJuangStepPattern(6, "c"))

    rebuilt_vec = np.zeros(len(alignment.index2))
    for index, val in enumerate(alignment.index2):
        rebuilt_vec[index] = rotation_full[i][val]
    
    rotation_full_aligned.append(rebuilt_vec)

for i in range(len(acceleration_full)):
    alignment = dtw(acceleration_full[3],acceleration_full[i], keep_internals=True, 
    step_pattern=rabinerJuangStepPattern(6, "c"))

    rebuilt_vec = np.zeros(len(alignment.index2))
    for index, val in enumerate(alignment.index2):
        rebuilt_vec[index] = acceleration_full[i][val]
    
    acceleration_full_aligned.append(rebuilt_vec)

acceleration_full_aligned = np.array(acceleration_full_aligned)
rotation_full_aligned = np.array(rotation_full_aligned)
for i in range(len(acceleration_full_aligned)):
    ax[0].plot(smooth(rotation_full_aligned[i]), c='r')
    ax[1].plot(smooth(acceleration_full_aligned[i]), c='r')

cmap = plt.cm.coolwarm
from matplotlib.lines import Line2D

custom_lines = [Line2D([0], [0], color=cmap(0.), lw=4),
                Line2D([0], [0], color=cmap(1.), lw=4)]

ax[0].legend(custom_lines, ['Baseline', 'Controlled'], fontsize=14, loc='lower right')
ax[1].legend(custom_lines, ['Baseline', 'Controlled'], fontsize=14, loc='lower right')


ax[0].set_ylabel("rotation", fontsize=20)
ax[1].set_ylabel("acceleration", fontsize=20)
ax[0].xaxis.set_tick_params(labelsize=12)
ax[1].xaxis.set_tick_params(labelsize=12)
ax[0].yaxis.set_tick_params(labelsize=12)
ax[1].yaxis.set_tick_params(labelsize=12)
ax[0].grid()
ax[1].grid()
plt.show()