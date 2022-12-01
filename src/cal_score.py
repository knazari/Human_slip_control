import glob
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
import matplotlib as mpl
import seaborn as sns

# def sigmoid_time(x):
#   return -1 / (1 + math.exp(-(x-50)/14)) + 1

# def sigmoid_rot(x):
#   return -1 / (1 + math.exp(-(x-20)/2.5)) + 1

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

mean_time = 80
scale_suc_time = 5
scale_fail_time = 30

mean_rot = 20
scale_suc_rot = 1.3
scale_fail_rot = 8

subject_baseline_data_path = "/home/kia/catkin_ws/src/data_collection_human_test/data/pilot_test2/subject_002/baseline"
subject_controlled_data_path = "/home/kia/catkin_ws/src/data_collection_human_test/data/pilot_test2/subject_002/controlled"

Baseline_files = sorted(glob.glob(subject_baseline_data_path + '/*'))
controlled_files = sorted(glob.glob(subject_controlled_data_path + '/*'))

S_time_full = []
S_rot_full = []
S_full_full = []

baseline_task_time = []
for file in Baseline_files:
    meta_data = pd.read_csv(file + '/meta_data.csv')
    baseline_task_time.append(meta_data['task_completion_time'][0])
    # print(meta_data['task_completion_time'][0])

mean_baseline_time = sum(baseline_task_time)/len(baseline_task_time)

for file in controlled_files:
    # for file in controlled_files:
    last_task_meta = pd.read_csv(file + '/meta_data.csv')
    task_time      = last_task_meta['task_completion_time'][0]
    task_start     = last_task_meta['start_index'][0]
    task_end       = last_task_meta['end_index'][0]

    object_imu     = pd.read_csv(file + '/object_imu.csv')
    object_imu_rot = R.from_quat(object_imu[['q1', 'q2', 'q3', 'q0']]).as_euler('zyx', degrees=True)
    absolute_rotation = object_imu_rot[:, 0] - object_imu_rot[0, 0]
    task_max_rotation = max(abs(absolute_rotation[int(task_start):int(task_end)]))

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

    def final_score(s1, s2):
        scaling_factor = sigmoid_scaling(s2, 0.5, 0.1)
        s = 2 * scaling_factor * ((((s1 * 100) ** 1.8*s2) / 100) / 40 + 0.2)
        clip_indeces = np.where(s < 1.0, s, 1)

        return clip_indeces

    Full_score = final_score(S_1_time, S_2_rotation)

    # print("S1: {:.2f}".format(S_1_time))
    # print("S2: {:.2f}".format(S_2_rotation))
    # print("S: {:.2f}".format(Full_score))
    # print("________")

    S_time_full.append(S_1_time)
    S_rot_full.append(S_2_rotation)
    S_full_full.append(Full_score)

    # # plt.plot(absolute_rotation)

    # rot_bar_color = ['red', 'orange', 'yellow', 'forestgreen']
    # fig, ax = plt.subplots(1, 3, figsize=(9,5.5), facecolor='lightskyblue')
    # fig.tight_layout(pad=8.0)
    # ax[0].bar(['Time score'], subject_time_score, color='limegreen', width=0.04)
    # ax[0].set_yticks([1,2,3,4])
    # ax[0].set_yticklabels(['Slow',  'Normal', 'Fast', 'Excellent'], fontsize=20)
    # ax[0].set_xticklabels(['Time score'], fontsize=20)
    # ax[0].set_title("Time scores", fontsize=30, pad=40)
    # ax2 = ax[0].twinx()
    # ax2.axhline(y=S_1_time, ls='-', color='black', linewidth=3)
    # ax2.set_yticks([0.25,0.5,0.75,1])
    # ax2.set_yticklabels(['0.25',  '0.5', '0.75', '1'], fontsize=20)

    # ax[1].bar(['Rotation score'], subject_rotation_score, width=0.04, color=rot_bar_color[subject_rotation_score-1])
    # ax[1].set_yticks([1,2,3,4])
    # ax[1].set_yticklabels(['','','', ''], fontsize=20, rotation=90)
    # ax[1].set_xticklabels(['Rotation score'], fontsize=20)
    # ax[1].set_title("Rotation score", fontsize=30, pad=40)
    # ax[1].axhline(y=2, ls='-', color='g', linewidth=0.5)
    # ax3 = ax[1].twinx()
    # ax3.axhline(y=S_2_rotation, ls='-', color='black', linewidth=3)
    # ax3.set_yticks([0.25,0.5,0.75,1])
    # ax3.set_yticklabels(['0.25',  '0.5', '0.75', '1'], fontsize=20)
    # ax[1].tick_params(axis='y', top=True)
    # ax[1].set_ylabel("Not Acceptable               Acceptable", fontsize=20)
    # ax3.text(-0.0045, S_2_rotation+0.02, str(int(task_max_rotation)), fontsize=10)

    # # ax[2].bar(['Full score'], Full_score, color=(1-Full_score, Full_score, 0), width=0.4)
    # ax[2].bar(['Full score'], Full_score, color='b', width=0.4)
    # ax[2].set_yticks([0.0, 0.25, 0.5, 0.75, 1.0])
    # ax[2].set_yticklabels(['','','', '', ''], fontsize=20, rotation=90)
    # ax[2].set_xticklabels(['Full score'], fontsize=20)
    # ax[2].set_title("Full score", fontsize=30, pad=40)

    # # # colormap
    # cmap = plt.get_cmap('RdYlGn', 100)

    # # Normalizer
    # norm = mpl.colors.Normalize(vmin=0, vmax=1)
    # # creating ScalarMappable
    # sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    # sm.set_array([])
    # plt.colorbar(sm, ax=ax[2])


x = np.arange(1, 21, 2)
tick_label = [1, 2,3,4,5,6,7,8,9,10]
plt.bar(x, S_time_full, align='edge', width=-0.5, label='time score', edgecolor='black', color='blue', tick_label=tick_label)
plt.bar(x, S_rot_full, align='edge', width=0.5, label='rotation score', edgecolor='black', color='green')
x = np.arange(1.5, 21.5, 2)
plt.bar(x, S_full_full, align='edge', width=0.5, label='combined score', edgecolor='black', color='red')
plt.ylabel("Score value", fontsize=18, labelpad=8)
plt.xlabel("Trial number", fontsize=18, labelpad=8)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.legend(fontsize=14)

plt.show()
