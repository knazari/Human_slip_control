from cProfile import label
import glob
import numpy as np
import pandas as pd
from scipy import integrate
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R

def smooth(x,window_len=11,window='hanning'):
    s = np.r_[x[window_len-1:0:-1],x,x[-2:-window_len-1:-1]]
    #print(len(s))
    if window == 'flat': #moving average
        w=np.ones(window_len,'d')
    else:
        w=eval('np.'+window+'(window_len)')

    y=np.convolve(w/w.sum(),s,mode='valid')
    #return y
    return y[int((window_len/2-1)):-int((window_len/2))]


# test_data_path = "/home/kia/catkin_ws/src/data_collection_human_test/data/subject_002/data_sample_2022-11-06-16-11-50"

user_path = "/home/kia/catkin_ws/src/data_collection_human_test/data/subject_003/controlled"

files = glob.glob(user_path + "/*")
for index, file in enumerate(files):
    hand_imu   = pd.read_csv(file + '/hand_imu.csv')
    object_imu = pd.read_csv(file + '/object_imu.csv')

    object_imu_rot = R.from_quat(object_imu[['q1', 'q2', 'q3', 'q0']]).as_euler('xyz', degrees=True)
    
    task_quat = pd.read_csv(file + '/object_marker.csv')[['marker_quaternion_x', 'marker_quaternion_y', 'marker_quaternion_z', 'marker_quaternion_w']]
    task_quat = np.array(task_quat)

    while np.where(task_quat==np.array([10., 10., 10., 10.]))[0].any():
        task_quat[np.unique(np.where(task_quat==np.array([10., 10., 10., 10.]))[0])] = task_quat[np.unique(np.where(task_quat==np.array([10., 10., 10., 10.]))[0]) - 1]

    task_object_rotation  = R.from_quat(task_quat).as_euler('xyz', degrees=True)[:, 2]
    task_object_rotation -= task_object_rotation[0]
    task_object_rotation = smooth(task_object_rotation)
    
    # if index != 7:
    #     plt.plot(-task_object_rotation + task_object_rotation[0], label="object rotataion")

    a_y_smoothed = -smooth(np.array(hand_imu['acc_y']))
    x_axis = np.arange(len(a_y_smoothed))
    v_y = integrate.cumtrapz(a_y_smoothed, x_axis*0.017, initial=0)
    # if index != 6:
    #     plt.plot(v_y, label="v_y")
    # # plt.plot(a_y_smoothed-a_y_smoothed[0])

    plt.plot(smooth(object_imu_rot[:, 1] - object_imu_rot[0, 1]) - smooth(object_imu_rot[:, 1] - object_imu_rot[0, 1])[0])


plt.show()
# rot_label = ['x', 'y', 'z']
# for i in range(3):
#     plt.plot(object_imu_rot[:, i] - object_imu_rot[0, i], label=rot_label[i])
#     plt.legend()
#     plt.figure()


# for i in range(3):
#     plt.plot(object_marker_rot[:, i] - object_marker_rot[0, i], label=rot_label[i])
#     plt.legend()
#     plt.figure()

# plt.show()

# normal_force_col = ['txl1_z', 'txl2_z', 'txl3_z', 'txl4_z', 'txl5_z', 'txl6_z', 'txl7_z', 'txl8_z', 'txl9_z', 'txl10_z', 'txl11_z', 'txl12_z', 
#                     'txl13_z', 'txl14_z', 'txl15_z', 'txl16_z']




# dir_path = "/home/kiyanoush/Desktop/HumanTest/PilotTest1/data/mohammadreza/baseline"
# files = glob.glob(dir_path + "/*")

# for index, file in enumerate(files):
#     hand_imu   = pd.read_csv(file + '/object_imu.csv')
#     object_imu = pd.read_csv(file + '/hand_imu.csv')
#     xela_normal = pd.read_csv(file + '/xela.csv')[normal_force_col].sum(axis=1)
#     plt.plot(xela_normal - xela_normal[0])

#     # # plot object rotation data
#     # if index == 0:
#     #     plt.plot(smooth(-object_imu['pitch'] + object_imu['pitch'][0])[15:40] - smooth(-object_imu['pitch'] + object_imu['pitch'][0])[15], linewidth=2)
#     # if index == 1:
#     #     plt.plot(smooth(-object_imu['pitch'] + object_imu['pitch'][0])[24:45] - smooth(-object_imu['pitch'] + object_imu['pitch'][0])[24], linewidth=2)
#     # if index == 2:
#     #     plt.plot(smooth(-object_imu['pitch'] + object_imu['pitch'][0])[19:40] - smooth(-object_imu['pitch'] + object_imu['pitch'][0])[19], linewidth=2)
#     # if index == 3:
#     #     plt.plot(smooth(-object_imu['pitch'] + object_imu['pitch'][0])[22:45] - smooth(-object_imu['pitch'] + object_imu['pitch'][0])[22], linewidth=2)
#     # if index == 4:
#     #     plt.plot(smooth(-object_imu['pitch'] + object_imu['pitch'][0])[40:60] - smooth(-object_imu['pitch'] + object_imu['pitch'][0])[40], linewidth=2)
#     # if index == 5:
#     #     plt.plot(smooth(-object_imu['pitch'] + object_imu['pitch'][0])[30:50] - smooth(-object_imu['pitch'] + object_imu['pitch'][0])[30], linewidth=2)
#     # if index == 6:
#     #     plt.plot(smooth(-object_imu['pitch'] + object_imu['pitch'][0])[39:59] - smooth(-object_imu['pitch'] + object_imu['pitch'][0])[39], linewidth=2)
#     # if index == 7:
#     #     plt.plot(smooth(-object_imu['pitch'] + object_imu['pitch'][0])[24:44] - smooth(-object_imu['pitch'] + object_imu['pitch'][0])[24], linewidth=2)
#     # if index == 8:
#     #     plt.plot(smooth(-object_imu['pitch'] + object_imu['pitch'][0])[26:48] - smooth(-object_imu['pitch'] + object_imu['pitch'][0])[26], linewidth=2)
#     # if index == 9:
#     #     plt.plot(smooth(-object_imu['pitch'] + object_imu['pitch'][0])[24:44] - smooth(-object_imu['pitch'] + object_imu['pitch'][0])[24], linewidth=2)


# plt.title("User 1 controlled mode Hand velocity", fontsize=20)
# plt.xticks(fontsize=16)
# plt.yticks(fontsize=16)
# plt.show()
# plt.savefig(dir_path + '/vel_plot.png')
