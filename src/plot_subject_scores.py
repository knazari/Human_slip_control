import glob
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
import matplotlib as mpl
import seaborn as sns

plt.rcParams["figure.figsize"] = (9,6.1)

def sigmoid_time(x, mean, scale_suc, scale_fail):
    if x <= mean:
        x = -1 / (1 + math.exp(-(x-mean)/scale_suc)) + 1
    else:
        x = -1 / (1 + math.exp(-(x-mean)/scale_fail)) + 1
    return x

def sigmoid_rot(x, mean_, scale_suc):
    x = -1 / (1 + math.exp(-(x-mean_)/scale_suc)) + 1
    return x

def sigmoid_scaling(x, mean_, scale_suc):
    x = 1 / (1 + math.exp(-(x-mean_)/scale_suc))
    return x

def final_score(s1, s2):
    scaling_factor = sigmoid_scaling(s2, 0.5, 0.1)
    s = 2 * scaling_factor * ((((s1 * 100) ** 1.8*s2) / 100) / 40 + 0.2)
    clip_indeces = np.where(s < 1.0, s, 1)

    return clip_indeces


subjects_to_test = ["subject_009", "subject_010", "subject_011", "subject_012", "subject_013", "subject_014", "subject_015", "subject_016", "subject_017", "subject_018", "subject_019", "subject_020", "subject_021", "subject_022", "subject_023", "subject_024"]

subjects_fam_mean_time_score = []
subjects_mean_time_score = []
subjects_fam_mean_combined_score = []
subjects_mean_combined_score = []

subjects_fam_num_time_score_larger2 = []
subjects_num_time_score_larger2 = []
subjects_fam_num_combined_score_larger2 = []
subjects_num_combined_score_larger2 = []

subjects_fam_max_time_score = []
subjects_max_time_score = []
subjects_fam_max_combined_score = []
subjects_max_combined_score = []

all_subjects_fam_time_score = []
all_subjects_fam_combined_score = []
all_subjects_time_score = []
all_subjects_combined_score = []

for name in subjects_to_test:
    print(name)
    save_location = "/home/kia/catkin_ws/src/data_collection_human_test/data/real_test2/plots/"

    subject_baseline_data_path_1       = "/home/kia/catkin_ws/src/data_collection_human_test/data/real_test2/" + name + "/baseline"
    subject_controlled_data_path_1     = "/home/kia/catkin_ws/src/data_collection_human_test/data/real_test2/" + name + "/controlled"
    subject_baseline_fam_data_path_1   = "/home/kia/catkin_ws/src/data_collection_human_test/data/real_test2/" + name + "/baseline_familiarization"
    subject_controlled_fam_data_path_1 = "/home/kia/catkin_ws/src/data_collection_human_test/data/real_test2/" + name + "/controlled_familiarization"
    
    mean_time = 0.95
    scale_suc_time = 0.3
    scale_fail_time = 0.3
    baseline = [subject_baseline_data_path_1, subject_baseline_fam_data_path_1]

    for index, subject_baseline_data_path in enumerate(baseline):
        Baseline_files = sorted(glob.glob(subject_baseline_data_path + '/*'))
        S_time_full = []

        baseline_task_time = []
        baseline_task_score = []
        for file in Baseline_files:
            meta_data = pd.read_csv(file + '/meta_data.csv')
            task_time = meta_data['task_completion_time'][0]

            if task_time <= 0.64:
                subject_time_score = 1.0
            elif 0.64 < task_time <= 0.96:
                subject_time_score = 0.75
            elif 0.96 < task_time <= 1.2:
                subject_time_score = 0.5
            elif task_time > 1.2:
                subject_time_score = 0.25

            baseline_task_time.append(task_time)
            baseline_task_score.append(subject_time_score)

        # # 1. plot each subject score
        # fig, ax1 = plt.subplots(figsize=(9, 6))
        
        # ax1.set_ylim(0, 1);
        # x = np.arange(1, (len(baseline_task_time)*2) + 1, 2)
        # tick_label = [i + 1 for i in range(len(baseline_task_time))]
        # ax1.bar(x, baseline_task_score, align='edge', width=-0.5, label='time score', edgecolor='black', color='red')
        # ax1.set_xlabel("Trial number", fontsize=22, labelpad=5)
        # ax1.set_ylabel("Time score", fontsize=22, labelpad=5)
        # ax1.set_xticklabels(tick_label,fontsize=18)
        # ax1.set_yticks([0.25, 0.5, 0.75, 1])
        # ax1.set_yticklabels(["Slow", "Normal", "Fast", "Excellent"], fontsize=16, rotation=75)
        # ax1.tick_params(axis='y', which='both', color='r', width=2.5, direction='in', length=6)
        # if index == 0: title = "baseline"
        # elif index == 1: title = "baseline_familirization"
        # ax1.set_title(title, fontsize=18)
        
        # ax2 = ax1.twinx()
        # ax2.bar(x, baseline_task_time, align='edge', width=0.5, label='time value', edgecolor='black', color='blue', tick_label=tick_label)
        # ax2.set_ylabel("Time value", fontsize=22, labelpad=5)
        # ax2.set_yticks([0.5, 1, 1.5, 2, 2.5, 3])
        # ax2.set_yticklabels([0.5, 1, 1.5, 2, 2.5, 3], fontsize=16)
        # ax2.tick_params(axis='y', which='both', color='blue', width=2.5, direction='in', length=6)
       
        # lines, labels = ax1.get_legend_handles_labels()
        # lines2, labels2 = ax2.get_legend_handles_labels()
        # ax2.legend(lines + lines2, labels + labels2, loc=0, fontsize=16)
        # plt.savefig(save_location + name + "__" + title + ".png")

        # 2. plot fam vs test mean scores and 4. geneder wise performance
        if index == 0:
            baseline_task_score = np.array(baseline_task_score)
            subjects_mean_time_score.append(np.mean(baseline_task_score))
            subjects_num_time_score_larger2.append(len(np.where(baseline_task_score>0.25)[0]))
            subjects_max_time_score.append(np.max(baseline_task_score))
        elif index == 1:
            baseline_task_score = np.array(baseline_task_score)
            subjects_fam_mean_time_score.append(np.mean(baseline_task_score))
            subjects_fam_num_time_score_larger2.append(len(np.where(baseline_task_score>0.25)[0]))
            subjects_fam_max_time_score.append(np.max(baseline_task_score))
        
        # 3. plot learning curve for familiarization
        if index == 1:
            all_subjects_fam_time_score.append(baseline_task_score)
        elif index == 0:
            all_subjects_time_score.append(baseline_task_score)

    ###########################################################
    ###########################################################
    ###########################################################
    mean_time = 60
    scale_suc_time = 5
    scale_fail_time = 20

    mean_rot = 15
    scale_suc_rot = 1.3
    scale_fail_rot = 8

    baseline = subject_baseline_data_path_1
    controlled = [subject_controlled_data_path_1, subject_controlled_fam_data_path_1]

    for index, subject_controlled_data_path in enumerate(controlled):

        Baseline_files = sorted(glob.glob(baseline + '/*'))
        controlled_files = sorted(glob.glob(subject_controlled_data_path + '/*'))

        S_time_full = []
        S_rot_full = []
        S_full_full = []

        baseline_task_time = []
        for file in Baseline_files:
            meta_data = pd.read_csv(file + '/meta_data.csv')
            baseline_task_time.append(meta_data['task_completion_time'][0])

        mean_baseline_time = sum(baseline_task_time)/len(baseline_task_time)

        for file in controlled_files:
            last_task_meta = pd.read_csv(file + '/meta_data.csv')
            task_time      = last_task_meta['task_completion_time'][0]
            task_start     = last_task_meta['start_index'][0]
            task_end       = last_task_meta['end_index'][0]
            object_imu     = pd.read_csv(file + '/object_imu.csv')
            object_imu_rot = R.from_quat(object_imu[['q1', 'q2', 'q3', 'q0']]).as_euler('xyx', degrees=True)
            absolute_rotation = object_imu_rot[:, 1] - object_imu_rot[0, 1]
            task_max_rotation = max(abs(absolute_rotation[int(task_start):int(task_end)]))

            time_increase_perc = ((task_time - mean_baseline_time) / mean_baseline_time) * 100

            S_1_time     = sigmoid_time(time_increase_perc, mean_time, scale_suc_time, scale_fail_time)
            S_2_rotation = sigmoid_rot(task_max_rotation, mean_rot, scale_suc_rot)
            Full_score   = final_score(S_1_time, S_2_rotation)

            S_time_full.append(S_1_time)
            S_rot_full.append(S_2_rotation)
            S_full_full.append(Full_score)

        # # 1. plot each subject score
        # plt.figure()
        # x = np.arange(1, (len(S_time_full)*2) + 1, 2)
        # tick_label = [i + 1 for i in range(len(S_time_full))]
        # plt.bar(x, S_time_full, align='edge', width=-0.5, label='time score', edgecolor='black', color='red')
        # plt.bar(x, S_rot_full, align='edge', width=0.5, label='rotation score', edgecolor='black', color='blue', tick_label=tick_label)
        # x = np.arange(1.5, (len(S_time_full)*2) + 1.5, 2)
        # plt.bar(x, S_full_full, align='edge', width=0.5, label='combined score', edgecolor='black', color='green')
        # plt.ylabel("Score value", fontsize=22, labelpad=8)
        # plt.xlabel("Trial number", fontsize=22, labelpad=8)
        # if index == 0: title = "controlled"
        # elif index == 1: title = "controlled_familirization"
        # plt.title(title, fontsize=18)
        # plt.xticks(fontsize=18)
        # plt.yticks(fontsize=18)
        # plt.legend(fontsize=18)
        # plt.savefig(save_location + name + "__" + title + ".png")

        # 2. plot fam vs test mean scores  and 4. geneder wise performance
        if index == 0:
            S_full_full = np.array(S_full_full)
            subjects_mean_combined_score.append(np.mean(S_full_full))
            subjects_num_combined_score_larger2.append(len(np.where(S_full_full>0.25)[0]))
            subjects_max_combined_score.append(np.max(S_full_full))
        elif index == 1:
            S_full_full = np.array(S_full_full)
            subjects_fam_mean_combined_score.append(np.mean(S_full_full))
            subjects_fam_num_combined_score_larger2.append(len(np.where(S_full_full>0.25)[0]))
            subjects_fam_max_combined_score.append(np.max(S_full_full))

        # 3. plot learning curve for familiarization
        if index == 1:
            all_subjects_fam_combined_score.append(S_full_full)
        elif index == 0:
            all_subjects_combined_score.append(S_full_full)     
    
    # plt.figure()


# # 2. plot fam vs test mean scores
# plt.figure()
# x = np.arange(1, (len(subjects_fam_mean_time_score)*2) + 1, 2)
# tick_label = [i + 1 for i in range(len(subjects_fam_mean_time_score))]
# plt.bar(x, subjects_fam_mean_time_score, align='edge', width=-0.5, label='Familiarization mean score', edgecolor='black', color='blue', tick_label=tick_label)
# plt.bar(x, subjects_mean_time_score, align='edge', width=0.5, label='Test mean score', edgecolor='black', color='red')
# # plt.bar(x, subjects_fam_num_time_score_larger2, align='edge', width=-0.5, label='Familiarization', edgecolor='black', color='blue', tick_label=tick_label)
# # plt.bar(x, subjects_num_time_score_larger2, align='edge', width=0.5, label='Test', edgecolor='black', color='red')
# plt.ylabel("Score value", fontsize=22, labelpad=8)
# # plt.ylabel("Number of trials", fontsize=22, labelpad=8)
# plt.xlabel("Subject number", fontsize=22, labelpad=8)
# plt.title("Baseline: Time score in Familiarization vs Test", fontsize=18)
# # plt.title("Baseline: number of trials with time score > 0.25", fontsize=18)
# plt.xticks(fontsize=18)
# plt.yticks(fontsize=18)
# plt.legend(fontsize=18)
# plt.figure()
# x = np.arange(1, (len(subjects_fam_mean_combined_score)*2) + 1, 2)
# tick_label = [i + 1 for i in range(len(subjects_fam_mean_combined_score))]
# plt.bar(x, subjects_fam_mean_combined_score, align='edge', width=-0.5, label='Familiarization mean score', edgecolor='black', color='blue', tick_label=tick_label)
# plt.bar(x, subjects_mean_combined_score, align='edge', width=0.5, label='Test mean score', edgecolor='black', color='red')
# # plt.bar(x, subjects_fam_num_combined_score_larger2, align='edge', width=-0.5, label='Familiarization', edgecolor='black', color='blue', tick_label=tick_label)
# # plt.bar(x, subjects_num_combined_score_larger2, align='edge', width=0.5, label='Test', edgecolor='black', color='red')
# plt.ylabel("Score value", fontsize=22, labelpad=8)
# # plt.ylabel("Number of trials", fontsize=22, labelpad=8)
# plt.xlabel("Subject number", fontsize=22, labelpad=8)
# plt.title("Controlled: Combined score in Familiarization vs Test", fontsize=18)
# # plt.title("Controlled: number of trials with combined score > 0.2", fontsize=18)
# plt.xticks(fontsize=18)
# plt.yticks(fontsize=18)
# plt.legend(fontsize=18)
# plt.show()

# 3. plot learning curve for familiarization and tests
all_subjects_combined_score = np.array(all_subjects_combined_score)#[[1, 6, 8, 9, 12, 13, 14, 15]] # the indeces are for choosing male/female data
plt.plot(np.arange(1, 11), np.mean(all_subjects_combined_score, axis=0), c='b', linewidth=2)
plt.xticks(np.arange(1, 11))
plt.ylabel("combined score", fontsize=22, labelpad=8)
plt.xlabel("Trial number", fontsize=22, labelpad=8)
plt.title("Average combined score of 16 subjects in baseline test trials", fontsize=22, pad=20)
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
# plt.show()

# 3.1 calculating improving performance on the last trials
first_4trial  = np.mean(all_subjects_combined_score[:, :7], axis=1)
# middle_3trial = np.mean(all_subjects_combined_score[:, 4:7], axis=1)
last_3trial   = np.mean(all_subjects_combined_score[:, 7:], axis=1)
print(np.where(last_3trial>first_4trial))
# print(np.where(last_3trial>middle_3trial))

# # 4. gender wise performance
# subjects_fam_mean_time_score = np.array(subjects_fam_mean_time_score)
# subjects_mean_time_score = np.array(subjects_mean_time_score)
# subjects_fam_mean_combined_score = np.array(subjects_fam_mean_combined_score)
# subjects_mean_combined_score = np.array(subjects_mean_combined_score)

# reorder_index = [0, 2, 3, 4, 5, 7, 10, 11, 1, 6, 8, 9, 12, 13, 14, 15]
# bar_colors = ['r', 'r', 'r', 'r', 'r', 'r', 'r', 'r', 'b', 'b', 'b', 'b', 'b', 'b', 'b', 'b']
# bar_tick_labels = ['F1', 'F2', 'F3', 'F4', 'F5', 'F6', 'F7', 'F8', 'M1', 'M2', 'M3', 'M4', 'M5', 'M6', 'M7', 'M8']

# x = np.arange(1, (len(subjects_fam_mean_time_score)*2) + 1, 2)
# plt.bar(x, subjects_fam_mean_time_score[reorder_index], align='edge', width=-0.5, label='Familiarization mean score', edgecolor='black', color=bar_colors, alpha=0.6, tick_label=bar_tick_labels)
# plt.bar(x, subjects_mean_time_score[reorder_index], align='edge', width=0.5, label='Test mean score', edgecolor='black', color=bar_colors)
# plt.axvline(x=16, c='black', ls='--')
# plt.ylabel("Score value", fontsize=22, labelpad=8)
# plt.xlabel("Subject number", fontsize=22, labelpad=8)
# plt.title("Baseline: Female vs Male scores in Baseline", fontsize=18)
# plt.xticks(fontsize=14)
# plt.yticks(fontsize=14)
# from matplotlib.lines import Line2D
# cmap = plt.cm.coolwarm
# custom_lines = [Line2D([0], [0], color=cmap(0.95), lw=6),
#                 Line2D([0], [0], color=cmap(0.), lw=6)]
# plt.legend(custom_lines, ['Female', 'Male'], fontsize=18)
# plt.figure()
# x = np.arange(1, (len(subjects_fam_mean_combined_score)*2) + 1, 2)
# plt.bar(x, subjects_fam_mean_combined_score[reorder_index], align='edge', width=-0.5, label='Familiarization mean score', edgecolor='black', color=bar_colors, alpha=0.6, tick_label=bar_tick_labels)
# plt.bar(x, subjects_mean_combined_score[reorder_index], align='edge', width=0.5, label='Test mean score', edgecolor='black', color=bar_colors)
# plt.axvline(x=16, c='black', ls='--')
# plt.ylabel("Score value", fontsize=22, labelpad=8)
# plt.xlabel("Subject number", fontsize=22, labelpad=8)
# plt.title("Controlled: Female vs Male scores in Controlled", fontsize=18)
# plt.xticks(fontsize=14)
# plt.yticks(fontsize=14)
# plt.legend(custom_lines, ['Female', 'Male'], fontsize=18)
# plt.show()
