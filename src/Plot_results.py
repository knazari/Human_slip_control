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


user_path = "/home/kia/catkin_ws/src/data_collection_human_test/data/subject_004/controlled"

files = glob.glob(user_path + "/*")
for index, file in enumerate(files):

    hand_imu       = pd.read_csv(file + '/hand_imu.csv')
    object_imu     = pd.read_csv(file + '/object_imu.csv')
    object_imu_rot = R.from_quat(object_imu[['q1', 'q2', 'q3', 'q0']]).as_euler('zyx', degrees=True)
    
    marker_quat = pd.read_csv(file + '/object_marker.csv')[['marker_quaternion_x', 'marker_quaternion_y', 'marker_quaternion_z', 'marker_quaternion_w']]
    marker_quat = np.array(marker_quat)

    # replace missed frames with last available value
    while np.where(marker_quat==np.array([10., 10., 10., 10.]))[0].any():
        marker_quat[np.unique(np.where(marker_quat==np.array([10., 10., 10., 10.]))[0])] = marker_quat[np.unique(np.where(marker_quat==np.array([10., 10., 10., 10.]))[0]) - 1]

    task_object_rotation  = R.from_quat(marker_quat).as_euler('xyz', degrees=True)[:, 2]
    task_object_rotation -= task_object_rotation[0]
    task_object_rotation  = smooth(task_object_rotation)
    
    # Plot marker rotation
    plt.plot(-task_object_rotation + task_object_rotation[0], label="object rotataion")

    # Plot object_imu rotation
    smooth_rotation = smooth(object_imu_rot[:, 0] - object_imu_rot[0, 0])
    # plt.plot(smooth_rotation - smooth_rotation[0])

    a_y_smoothed = -smooth(np.array(hand_imu['acc_y']))
    x_axis = np.arange(len(a_y_smoothed))
    v_y = integrate.cumtrapz(a_y_smoothed, x_axis*0.017, initial=0)

    # plt.plot(v_y, label="v_y")
    # # plt.plot(a_y_smoothed-a_y_smoothed[0])


plt.show()
