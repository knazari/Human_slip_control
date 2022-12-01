import glob
import time
import math
import numpy as np
import pandas as pd
from scipy import integrate
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
from dtw import *


def sigmoid_time(x, mean, scale_suc, scale_fail):
    if x <= mean:
        x = -1 / (1 + math.exp(-(x-mean)/scale_suc)) + 1
    else:
        x = -1 / (1 + math.exp(-(x-mean)/scale_fail)) + 1
    return x
  
#   return -1 / (1 + math.exp(-(x-50)/14)) + 1

def sigmoid_rot(x, mean_, scale_suc):
    #   if x <= mean_:
    #     x = -1 / (1 + math.exp(-(x-mean_)/scale_suc)) + 1
    #   else:
    #     x = -1 / (1 + math.exp(-(x-mean_)/scale_fail)) + 1
    x = -1 / (1 + math.exp(-(x-mean_)/scale_suc)) + 1
    return x

def sigmoid_scaling(x, mean_, scale_suc):
    
    x = 1 / (1 + math.exp(-(x-mean_)/scale_suc)) 
    return x

def final_score(s1, s2):
    scaling_factor = sigmoid_scaling(s2, 0.5, 0.1)
    s = 2 * (scaling_factor* ((((s1 * 100) ** 1.8*s2) / 100) / 40 + 0.2))
    clip_indeces = np.where(s < 1.0, s, 1)

    return clip_indeces

mean_time = 60
scale_suc_time = 5
scale_fail_time = 30

mean_rot = 20
scale_suc_rot = 1
scale_fail_rot = 8


subject_baseline_data_path = "/home/kia/catkin_ws/src/data_collection_human_test/data/pilot_test2/subject_001/baseline"
subject_controlled_data_path = "/home/kia/catkin_ws/src/data_collection_human_test/data/pilot_test2/subject_001/controlled"

Baseline_files = sorted(glob.glob(subject_baseline_data_path + '/*'))
controlled_files = sorted(glob.glob(subject_controlled_data_path + '/*'))

baseline_task_time = []
for index, file in enumerate(Baseline_files):
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
    
    baseline_task_time.append((end_index-start_index)/58.)
    # print(meta_data['task_completion_time'][0])

mean_baseline_time = sum(baseline_task_time)/len(baseline_task_time)
Full_score_list = []

for file in controlled_files:
    # for file in controlled_files:
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
    
    task_time = (start_index/end_index) / 58.

    object_imu     = pd.read_csv(file + '/object_imu.csv')
    object_imu_rot = R.from_quat(object_imu[['q1', 'q2', 'q3', 'q0']]).as_euler('zyx', degrees=True)
    absolute_rotation = object_imu_rot[start_index:end_index, 0] - object_imu_rot[start_index, 0]
    task_max_rotation = max(abs(absolute_rotation))

    # calculate task completion time socre:
    # score bands: 
    # < 25%   increase  : Excellent
    # 25-50%  increase  : Fast
    # 50-75%  increase  : Good
    # >75%    increase  : Slow

    time_increase_perc = ((task_time - mean_baseline_time) / mean_baseline_time) * 100

    if time_increase_perc <= 30:
        subject_time_score = 4
    elif 30 < time_increase_perc <= 60:
        subject_time_score = 3
    elif 60 < time_increase_perc <= 90:
        subject_time_score = 2
    elif time_increase_perc > 90:
        subject_time_score = 1

    S_1_time = sigmoid_time(time_increase_perc, mean_time, scale_suc_time, scale_fail_time)

    # calculate object rotation socre:
    # score bands: 
    # <10   deg   : Excellent
    # 10-20 deg   : Good
    # 20-30 deg   : Fair
    # >30   deg   : Weak

    if task_max_rotation <= 10:
        subject_rotation_score = 4
    elif 10 < task_max_rotation <= 20:
        subject_rotation_score = 3
    elif 20 < task_max_rotation <= 30:
        subject_rotation_score = 2
    elif task_max_rotation > 30:
        subject_rotation_score = 1

    S_2_rotation = sigmoid_rot(task_max_rotation, mean_rot, scale_suc_rot)

    # Full_score = 2 * S_1_time * S_2_rotation / (S_1_time + S_2_rotation)
    w1 = 1
    w2 = 3
    # Full_score = (w1 * S_1_time + w2*S_2_rotation) / (w1 + w2)

    Full_score = final_score(S_1_time, S_2_rotation)

    Full_score_list.append(Full_score)



def smooth(x,window_len=11,window='hanning'):
    s = np.r_[x[window_len-1:0:-1],x,x[-2:-window_len-1:-1]]
    if window == 'flat': #moving average
        w=np.ones(window_len,'d')
    else:
        w=eval('np.'+window+'(window_len)')
    y=np.convolve(w/w.sum(),s,mode='valid')
    
    return y[int((window_len/2-1)):-int((window_len/2))]


# user_path = "/home/kia/catkin_ws/src/data_collection_human_test/data/pilot_test2/subject_001/baseline"
# files = sorted(glob.glob(user_path + "/*"))

# rotation_full     = []
# max_normal_force  = []
# acceleration_full = []
# fig, ax = plt.subplots(2)

# for index, file in enumerate(files):
#     if index == 0:
#         # start_index = 250
#         # end_index   = 300
#         continue
#     elif index ==1:
#         # start_index = 278
#         # end_index   = 342
#         continue
#     elif index ==2:
#         start_index = 480
#         end_index   = 580
#     elif index ==3:
#         start_index = 480
#         end_index   = 580
#     elif index ==4:
#         start_index = 430
#         end_index   = 530
#     elif index ==5:
#         start_index = 350
#         end_index   = 450
#     elif index ==6:
#         start_index = 350
#         end_index   = 450
#     elif index ==7:
#         start_index = 360
#         end_index   = 460
#     elif index ==8:
#         start_index = 375
#         end_index   = 475
#     elif index ==9:
#         start_index = 360
#         end_index   = 460
    
#     meta_data       = pd.read_csv(file + '/meta_data.csv')
#     hand_imu        = pd.read_csv(file + '/hand_imu.csv')
#     object_imu      = pd.read_csv(file + '/object_imu.csv')
#     xela_data       = pd.read_csv(file + '/xela.csv')[['txl1_z', 'txl2_z', 'txl3_z', 'txl4_z', 'txl5_z', 'txl6_z', 'txl7_z', 'txl8_z', 'txl9_z', 'txl10_z',
#                                                         'txl11_z', 'txl12_z', 'txl13_z', 'txl14_z', 'txl15_z', 'txl16_z']]
    
#     # start_index = meta_data['start_index'][0]
#     # end_index   = meta_data['end_index'][0]

#     # Object rotation
#     object_imu_rot  = R.from_quat(object_imu[['q1', 'q2', 'q3', 'q0']]).as_euler('zyx', degrees=True)
#     object_rotation = smooth(object_imu_rot[start_index:end_index, 0] - object_imu_rot[start_index, 0])
#     rotation_full.append(object_rotation - object_rotation[0])

#     # Hand acceleration
#     a_y_smoothed = -smooth(np.array(hand_imu['acc_y'])[start_index:end_index])
#     v_y = integrate.cumtrapz(a_y_smoothed, np.arange(len(a_y_smoothed))*0.017, initial=0)
#     acceleration = a_y_smoothed - a_y_smoothed[0]
#     acceleration_full.append(acceleration)
    
#     # Max grip force
#     xela_data = np.array(xela_data[start_index:end_index])
#     max_normal_force.append(np.max(np.sum(xela_data, axis=1)))

    
#     # fig, ax = plt.subplots(2)
#     # ax[0].plot(object_rotation, c='b')
#     # ax[1].plot(acceleration, c='b')

#     # ax[0].vlines(start_index, -max(abs(object_rotation)), max(abs(object_rotation)), color='c')
#     # ax[0].vlines(end_index, -max(abs(object_rotation)), max(abs(object_rotation)), color='c')
#     # ax[1].vlines(start_index, -max(abs(acceleration)), max(abs(acceleration)), color='c')
#     # ax[1].vlines(end_index, -max(abs(acceleration)), max(abs(acceleration)), color='c')

#     # ax[0].set_xticks(np.arange(0, len(object_rotation), 50))
#     # ax[1].set_xticks(np.arange(0, len(object_rotation), 50))

# rotation_full_aligned = []
# acceleration_full_aligned = []

# for i in range(len(rotation_full)):
#     alignment = dtw(rotation_full[1],rotation_full[i], keep_internals=True, 
#     step_pattern=rabinerJuangStepPattern(6, "c"))

#     rebuilt_vec = np.zeros(len(alignment.index2))
#     for index, val in enumerate(alignment.index2):
#         rebuilt_vec[index] = rotation_full[i][val]
    
#     rotation_full_aligned.append(rebuilt_vec)

# for i in range(len(acceleration_full)):
#     alignment = dtw(acceleration_full[3],acceleration_full[i], keep_internals=True, 
#     step_pattern=rabinerJuangStepPattern(6, "c"))

#     rebuilt_vec = np.zeros(len(alignment.index2))
#     for index, val in enumerate(alignment.index2):
#         rebuilt_vec[index] = acceleration_full[i][val]
    
#     acceleration_full_aligned.append(rebuilt_vec)

# acceleration_full_aligned = np.array(acceleration_full_aligned)
# rotation_full_aligned = np.array(rotation_full_aligned)
# for i in range(len(acceleration_full_aligned)):
#     ax[0].plot(smooth(rotation_full_aligned[i]), c='b')
#     ax[1].plot(smooth(acceleration_full_aligned[i]), c='b')

# user_path = "/home/kia/catkin_ws/src/data_collection_human_test/data/pilot_test2/subject_001/controlled"
# files = sorted(glob.glob(user_path + "/*"))

# rotation_full     = []
# max_normal_force  = []
# acceleration_full = []

# for index, file in enumerate(files):
#     if index == 0:
#         start_index = 380
#         end_index   = -25
#     elif index ==1:
#         start_index = 425
#         end_index   = -30
#     elif index ==2:
#         start_index = 400
#         end_index   = -10
#     elif index ==3:
#         start_index = 455
#         end_index   = -5
#     elif index ==4:
#         start_index = 400
#         end_index   = -1
#     elif index ==5:
#         start_index = 580
#         end_index   = -10
#     elif index ==6:
#         start_index = 380
#         end_index   = 550
#     elif index ==7:
#         start_index = 420
#         end_index   = -1
#     elif index ==8:
#         start_index = 430
#         end_index   = -1
#     elif index ==9:
#         start_index = 460
#         end_index   = -1
    
#     meta_data       = pd.read_csv(file + '/meta_data.csv')
#     hand_imu        = pd.read_csv(file + '/hand_imu.csv')
#     object_imu      = pd.read_csv(file + '/object_imu.csv')
#     xela_data       = pd.read_csv(file + '/xela.csv')[['txl1_z', 'txl2_z', 'txl3_z', 'txl4_z', 'txl5_z', 'txl6_z', 'txl7_z', 'txl8_z', 'txl9_z', 'txl10_z',
#                                                         'txl11_z', 'txl12_z', 'txl13_z', 'txl14_z', 'txl15_z', 'txl16_z']]
    
#     # start_index = meta_data['start_index'][0]
#     # end_index   = meta_data['end_index'][0]

#     # Object rotation
#     object_imu_rot  = R.from_quat(object_imu[['q1', 'q2', 'q3', 'q0']]).as_euler('zyx', degrees=True)
#     object_rotation = smooth(object_imu_rot[start_index:end_index, 0] - object_imu_rot[start_index, 0])
#     rotation_full.append(object_rotation - object_rotation[0])

#     # Hand acceleration
#     a_y_smoothed = -smooth(np.array(hand_imu['acc_y'])[start_index:end_index])
#     v_y = integrate.cumtrapz(a_y_smoothed, np.arange(len(a_y_smoothed))*0.017, initial=0)
#     acceleration = a_y_smoothed - a_y_smoothed[0]
#     acceleration_full.append(acceleration)
    
#     # Max grip force
#     xela_data = np.array(xela_data[start_index:end_index])
#     max_normal_force.append(np.max(np.sum(xela_data, axis=1)))

#     # ax[0].plot(object_rotation, c='r')
#     # ax[1].plot(acceleration, c='r')

#     # ax[0].set_xticks(np.arange(0, len(object_rotation), 50))
#     # ax[1].set_xticks(np.arange(0, len(object_rotation), 50))

# rotation_full_aligned = []
# acceleration_full_aligned = []

# for i in range(len(rotation_full)):
#     alignment = dtw(rotation_full[1],rotation_full[i], keep_internals=True, 
#     step_pattern=rabinerJuangStepPattern(6, "c"))

#     rebuilt_vec = np.zeros(len(alignment.index2))
#     for index, val in enumerate(alignment.index2):
#         rebuilt_vec[index] = rotation_full[i][val]
    
#     rotation_full_aligned.append(rebuilt_vec)

# for i in range(len(acceleration_full)):
#     alignment = dtw(acceleration_full[3],acceleration_full[i], keep_internals=True, 
#     step_pattern=rabinerJuangStepPattern(6, "c"))

#     rebuilt_vec = np.zeros(len(alignment.index2))
#     for index, val in enumerate(alignment.index2):
#         rebuilt_vec[index] = acceleration_full[i][val]
    
#     acceleration_full_aligned.append(rebuilt_vec)

# acceleration_full_aligned = np.array(acceleration_full_aligned)
# rotation_full_aligned = np.array(rotation_full_aligned)
# for i in range(len(acceleration_full_aligned)):
#     ax[0].plot(smooth(rotation_full_aligned[i]), c='r')
#     ax[1].plot(smooth(acceleration_full_aligned[i]), c='r')

# cmap = plt.cm.coolwarm
# from matplotlib.lines import Line2D

# custom_lines = [Line2D([0], [0], color=cmap(0.), lw=4),
#                 Line2D([0], [0], color=cmap(1.), lw=4)]

# ax[0].legend(custom_lines, ['Baseline', 'Controlled'], fontsize=14, loc='lower right')
# ax[1].legend(custom_lines, ['Baseline', 'Controlled'], fontsize=14, loc='lower right')


# ax[0].set_ylabel("rotation", fontsize=20)
# ax[1].set_ylabel("acceleration", fontsize=20)
# ax[0].xaxis.set_tick_params(labelsize=12)
# ax[1].xaxis.set_tick_params(labelsize=12)
# ax[0].yaxis.set_tick_params(labelsize=12)
# ax[1].yaxis.set_tick_params(labelsize=12)
# ax[0].grid()
# ax[1].grid()
# plt.show()