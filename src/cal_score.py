import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R


subject_baseline_data_path = "/home/kia/catkin_ws/src/data_collection_human_test/data/subject_004/baseline"
subject_controlled_data_path = "/home/kia/catkin_ws/src/data_collection_human_test/data/subject_004/controlled"

Baseline_files = sorted(glob.glob(subject_baseline_data_path + '/*'))
controlled_files = sorted(glob.glob(subject_controlled_data_path + '/*'))

baseline_task_time = []
for file in Baseline_files:
    meta_data = pd.read_csv(file + '/meta_data.csv')
    baseline_task_time.append(meta_data['task_completion_time'])

mean_baseline_time = sum(baseline_task_time)/len(baseline_task_time)

last_task_meta = pd.read_csv(controlled_files[-2] + '/meta_data.csv')
task_time  = last_task_meta['task_completion_time']
task_start = last_task_meta['start_index']
task_end   = last_task_meta['end_index']

# task_quat = pd.read_csv(controlled_files[-2] + '/object_marker.csv')[['marker_quaternion_x', 'marker_quaternion_y', 'marker_quaternion_z', 'marker_quaternion_w']]
# task_quat = np.array(task_quat)

# while np.where(task_quat==np.array([10., 10., 10., 10.]))[0].any():
#     task_quat[np.unique(np.where(task_quat==np.array([10., 10., 10., 10.]))[0])] = task_quat[np.unique(np.where(task_quat==np.array([10., 10., 10., 10.]))[0]) - 1]


# task_object_rotation  = R.from_quat(task_quat).as_euler('xyz', degrees=True)[:, 2]
# task_object_rotation -= task_object_rotation[0]
# task_max_rotation     = max(abs(task_object_rotation))


object_imu     = pd.read_csv(controlled_files[-2] + '/object_imu.csv')
object_imu_rot = R.from_quat(object_imu[['q1', 'q2', 'q3', 'q0']]).as_euler('zyx', degrees=True)
absolute_rotation = object_imu_rot[:, 0] - object_imu_rot[0, 0]
task_max_rotation = max(abs(absolute_rotation[int(task_start):int(task_end)]))

# calculate task completion time socre:
# score bands: 
# 10-20%  increase  : Excellent
# 20-30%  increase  : Good
# 30-40%  increase  : Fair
# 40-100% increase  : Weak


time_increase_perc = ((task_time - mean_baseline_time) / mean_baseline_time)[0] * 100
if time_increase_perc <= 20:
    subject_time_score = 4
elif 20 < time_increase_perc <= 30:
    subject_time_score = 3
elif 30 < time_increase_perc <= 40:
    subject_time_score = 2
elif time_increase_perc > 40:
    subject_time_score = 1


# calculate object rotation socre:
# score bands: 
# 0-6 deg  : Excellent
# 6-10%    : Good
# 10-15%   : Fair
# >15%     : Weak


if task_max_rotation <= 6:
    subject_rotation_score = 4
elif 6 < task_max_rotation <= 10:
    subject_rotation_score = 3
elif 10 < task_max_rotation <= 15:
    subject_rotation_score = 2
elif task_max_rotation > 15:
    subject_rotation_score = 1

print(subject_rotation_score)
plt.plot(absolute_rotation)
plt.figure()

plt.rcParams["figure.figsize"] = (20,10)
plt.bar(['Time score',  'Rotation score'], (subject_time_score, subject_rotation_score), color=['g', 'b'], width=0.6)
plt.yticks((1, 2, 3, 4), ('Weak',  'Fair', 'Good', 'Excellent'), fontsize=30)
plt.xticks(fontsize=30)
plt.title("Your task performance scores", fontsize=30)
plt.show()


    



