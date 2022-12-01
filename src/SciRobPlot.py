from cProfile import label
import glob
import numpy as np
import pandas as pd
from scipy import integrate
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
from matplotlib.ticker import FormatStrFormatter
import matplotlib.patches as patches

from dtw import *

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

plt.rcParams["figure.figsize"] = (9,5)

user_path = "/home/kia/catkin_ws/src/data_collection_human_test/data/subject_001/baseline"

max_normal_force = []
acceleration_full = []
rotation_full = []

files = sorted(glob.glob(user_path + "/*"))
for index, file in enumerate(files):

    if index == 0:
        # start_index = 250
        # end_index   = 300
        continue
    elif index ==1:
        start_index = 278
        end_index   = 342
    elif index ==2:
        start_index = 275
        end_index   = -1
    elif index ==3:
        start_index = 272
        end_index   = 310
    elif index ==4:
        start_index = 275
        end_index   = 320
    elif index ==5:
        start_index = 212
        end_index   = 279
    elif index ==6:
        start_index = 212
        end_index   = 276
    elif index ==7:
        start_index = 208
        end_index   = 271
    elif index ==8:
        start_index = 206
        end_index   = 271
    elif index ==9:
        # start_index = 225
        # end_index   = 290
        continue
    
    meta_data      = pd.read_csv(file + '/meta_data.csv')
    hand_imu       = pd.read_csv(file + '/hand_imu.csv')[start_index:end_index]
    object_imu     = pd.read_csv(file + '/object_imu.csv')[start_index:end_index]
    object_imu_rot = R.from_quat(object_imu[['q1', 'q2', 'q3', 'q0']]).as_euler('zyx', degrees=True)
    xela_data = pd.read_csv(file + '/xela.csv')[['txl1_z', 'txl2_z', 'txl3_z', 'txl4_z', 'txl5_z', 'txl6_z',
                                    'txl7_z', 'txl8_z', 'txl9_z', 'txl10_z', 'txl11_z', 'txl12_z', 'txl13_z',
                                    'txl14_z', 'txl15_z', 'txl16_z']]
    
    xela_data = np.array(xela_data[start_index:end_index])
    # xela_data = np.array(xela_data[204:304])
    
    max_normal_force.append(np.max(np.sum(xela_data, axis=1)))

    # Plot object_imu rotation
    smooth_rotation = smooth(object_imu_rot[:, 0] - object_imu_rot[0, 0])
    # plt.plot(smooth_rotation - smooth_rotation[0])
    rotation_full.append(smooth_rotation - smooth_rotation[0])

    a_y_smoothed = -smooth(np.array(hand_imu['acc_y']))
    x_axis = np.arange(len(a_y_smoothed))
    v_y = integrate.cumtrapz(a_y_smoothed, x_axis*0.017, initial=0)

    # plt.plot(v_y, label="v_y")
    # plt.plot(a_y_smoothed-a_y_smoothed[0])
    acceleration_full.append(a_y_smoothed-a_y_smoothed[0])

acceleration_full_aligned = []
for i in range(len(acceleration_full)):
    alignment = dtw(acceleration_full[3],acceleration_full[i], keep_internals=True, 
    step_pattern=rabinerJuangStepPattern(6, "c"))

    rebuilt_vec = np.zeros(len(alignment.index2))
    for index, val in enumerate(alignment.index2):
        rebuilt_vec[index] = acceleration_full[i][val]
    
    acceleration_full_aligned.append(rebuilt_vec)

max_normal_force = []
acceleration_full = []
rotation_full = []

for index, file in enumerate(files):

    start_index = 204
    end_index = 304

    if index != 0:
        meta_data      = pd.read_csv(file + '/meta_data.csv')
        hand_imu       = pd.read_csv(file + '/hand_imu.csv')[start_index:end_index]
        object_imu     = pd.read_csv(file + '/object_imu.csv')[start_index:end_index]
        object_imu_rot = R.from_quat(object_imu[['q1', 'q2', 'q3', 'q0']]).as_euler('zyx', degrees=True)
        xela_data = pd.read_csv(file + '/xela.csv')[['txl1_z', 'txl2_z', 'txl3_z', 'txl4_z', 'txl5_z', 'txl6_z',
                                        'txl7_z', 'txl8_z', 'txl9_z', 'txl10_z', 'txl11_z', 'txl12_z', 'txl13_z',
                                        'txl14_z', 'txl15_z', 'txl16_z']]
        
        xela_data = np.array(xela_data[start_index:end_index])
        # xela_data = np.array(xela_data[204:304])
        
        max_normal_force.append(np.max(np.sum(xela_data, axis=1)))
        
        marker_quat = pd.read_csv(file + '/object_marker.csv')[['marker_quaternion_x', 'marker_quaternion_y', 'marker_quaternion_z', 'marker_quaternion_w']]
        marker_quat = np.array(marker_quat)

        
        # Plot marker rotation
        # plt.plot(-task_object_rotation + task_object_rotation[0], label="object rotataion")

        # Plot object_imu rotation
        smooth_rotation = smooth(object_imu_rot[:, 0] - object_imu_rot[0, 0])
        # plt.plot(smooth_rotation - smooth_rotation[0])
        rotation_full.append(smooth_rotation - smooth_rotation[0])

        a_y_smoothed = -smooth(np.array(hand_imu['acc_y']))
        x_axis = np.arange(len(a_y_smoothed))
        v_y = integrate.cumtrapz(a_y_smoothed, x_axis*0.017, initial=0)

        # plt.plot(v_y, label="v_y")
        # plt.plot(a_y_smoothed-a_y_smoothed[0])
        acceleration_full.append(a_y_smoothed-a_y_smoothed[0])

rotation_full_aligned = []

# for i in range(len(acceleration_full_aligned)):
#     plt.plot(np.arange(len(smooth(acceleration_full_aligned[i])))/60, smooth(acceleration_full_aligned[i]), linewidth=1.8)

for i in range(len(rotation_full)):
    alignment = dtw(rotation_full[0],rotation_full[i], keep_internals=True, 
    step_pattern=rabinerJuangStepPattern(6, "c"))

    rebuilt_vec = np.zeros(len(alignment.index2))
    for index, val in enumerate(alignment.index2):
        rebuilt_vec[index] = rotation_full[i][val]
    
    rotation_full_aligned.append(rebuilt_vec)

# for i in range(len(rotation_full_aligned)):
#     # if i != 3:
#     plt.plot(smooth(rotation_full_aligned[i]), linewidth=1.8)


user_path = "/home/kia/catkin_ws/src/data_collection_human_test/data/subject_001/controlled"

max_normal_force = []
acceleration_full_cont = []
rotation_full_cont = []

files = sorted(glob.glob(user_path + "/*"))
for index, file in enumerate(files):

    if index == 0:
        start_index = 230
        end_index   = 370
    elif index ==1:
        start_index = 215
        end_index   = 336
    elif index ==2:
        start_index = 240
        end_index   = 330
    elif index ==3:
        start_index = 225
        end_index   = 371
    elif index ==4:
        start_index = 230
        end_index   = 350
    elif index ==5:
        start_index = 260
        end_index   = 370
    elif index ==6:
        start_index = 248
        end_index   = -3
    elif index ==7:
        start_index = 226
        end_index   = -3
    elif index ==8:
        start_index = 258
        end_index   = -3
    elif index ==9:
        start_index = 245
        end_index   = -1
    
    if index != 19:
        meta_data      = pd.read_csv(file + '/meta_data.csv')
        hand_imu       = pd.read_csv(file + '/hand_imu.csv')[start_index:end_index]
        object_imu     = pd.read_csv(file + '/object_imu.csv')[start_index:end_index]
        object_imu_rot = R.from_quat(object_imu[['q1', 'q2', 'q3', 'q0']]).as_euler('zyx', degrees=True)
        xela_data = pd.read_csv(file + '/xela.csv')[['txl1_z', 'txl2_z', 'txl3_z', 'txl4_z', 'txl5_z', 'txl6_z',
                                        'txl7_z', 'txl8_z', 'txl9_z', 'txl10_z', 'txl11_z', 'txl12_z', 'txl13_z',
                                        'txl14_z', 'txl15_z', 'txl16_z']]
        
        xela_data = np.array(xela_data[start_index:end_index])
        # xela_data = np.array(xela_data[204:304])
        
        max_normal_force.append(np.max(np.sum(xela_data, axis=1)))
        
        marker_quat = pd.read_csv(file + '/object_marker.csv')[['marker_quaternion_x', 'marker_quaternion_y', 'marker_quaternion_z', 'marker_quaternion_w']]
        marker_quat = np.array(marker_quat)

        
        # Plot marker rotation
        # plt.plot(-task_object_rotation + task_object_rotation[0], label="object rotataion")

        # Plot object_imu rotation
        smooth_rotation = smooth(object_imu_rot[:, 0] - object_imu_rot[0, 0])
        # plt.plot(smooth_rotation - smooth_rotation[0])
        rotation_full_cont.append(smooth_rotation - smooth_rotation[0])

        a_y_smoothed = -smooth(np.array(hand_imu['acc_y']))
        x_axis = np.arange(len(a_y_smoothed))
        v_y = integrate.cumtrapz(a_y_smoothed, x_axis*0.017, initial=0)

        # plt.plot(v_y, label="v_y")
        # plt.plot(a_y_smoothed-a_y_smoothed[0])
        acceleration_full_cont.append(a_y_smoothed-a_y_smoothed[0])


acceleration_full_cont_aligned = []
rotation_full_cont_aligned = []

# plt.figure()

for i in range(len(acceleration_full_cont)):
    alignment = dtw(acceleration_full_cont[1],acceleration_full_cont[i], keep_internals=True, 
    step_pattern=rabinerJuangStepPattern(6, "c"))

    rebuilt_vec = np.zeros(len(alignment.index2))
    for index, val in enumerate(alignment.index2):
        rebuilt_vec[index] = acceleration_full_cont[i][val]
    
    acceleration_full_cont_aligned.append(rebuilt_vec)

max_normal_force = []
acceleration_full_cont = []
rotation_full_cont = []

for index, file in enumerate(files):

    if index == 0:
        start_index = 195
        end_index   = 350
    elif index ==1:
        start_index = 190
        end_index   = 350
    elif index ==2:
        start_index = 200
        end_index   = 350
    elif index ==3:
        start_index = 190
        end_index   = 370
    elif index ==4:
        start_index = 195
        end_index   = -2
    elif index ==5:
        start_index = 218
        end_index   = -1
    elif index ==6:
        start_index = 200
        end_index   = -3
    elif index ==7:
        start_index = 185
        end_index   = -1
    elif index ==8:
        # start_index = 225
        # end_index   = -1
        continue
    elif index ==9:
        # start_index = 225
        # end_index   = -1
        continue
    
    if index != 19:
        meta_data      = pd.read_csv(file + '/meta_data.csv')
        hand_imu       = pd.read_csv(file + '/hand_imu.csv')[start_index:end_index]
        object_imu     = pd.read_csv(file + '/object_imu.csv')[start_index:end_index]
        object_imu_rot = R.from_quat(object_imu[['q1', 'q2', 'q3', 'q0']]).as_euler('zyx', degrees=True)
        xela_data = pd.read_csv(file + '/xela.csv')[['txl1_z', 'txl2_z', 'txl3_z', 'txl4_z', 'txl5_z', 'txl6_z',
                                        'txl7_z', 'txl8_z', 'txl9_z', 'txl10_z', 'txl11_z', 'txl12_z', 'txl13_z',
                                        'txl14_z', 'txl15_z', 'txl16_z']]
        
        xela_data = np.array(xela_data[start_index:end_index])
        # xela_data = np.array(xela_data[204:304])
        
        max_normal_force.append(np.max(np.sum(xela_data, axis=1)))
        
        marker_quat = pd.read_csv(file + '/object_marker.csv')[['marker_quaternion_x', 'marker_quaternion_y', 'marker_quaternion_z', 'marker_quaternion_w']]
        marker_quat = np.array(marker_quat)

        
        # Plot marker rotation
        # plt.plot(-task_object_rotation + task_object_rotation[0], label="object rotataion")

        # Plot object_imu rotation
        smooth_rotation = smooth(object_imu_rot[:, 0] - object_imu_rot[0, 0])
        # plt.plot(smooth_rotation - smooth_rotation[0])
        rotation_full_cont.append(smooth_rotation - smooth_rotation[0])

        a_y_smoothed = -smooth(np.array(hand_imu['acc_y']))
        x_axis = np.arange(len(a_y_smoothed))
        v_y = integrate.cumtrapz(a_y_smoothed, x_axis*0.017, initial=0)

        # plt.plot(v_y, label="v_y")
        # plt.plot(a_y_smoothed-a_y_smoothed[0])
        acceleration_full_cont.append(a_y_smoothed-a_y_smoothed[0])


# for i in range(len(acceleration_full_cont_aligned)):
#     plt.plot(np.arange(len(smooth(acceleration_full_cont_aligned[i])))/60, smooth(acceleration_full_cont_aligned[i]), linewidth=1.8)

# plt.xlabel("time step (60hz)", fontsize=16)
# plt.ylabel("acceleration (m/s^2)", fontsize=18, labelpad=-1.5)
# plt.title("Subject hand acceleration", fontsize=20)
# plt.xticks(fontsize=14)
# plt.yticks(fontsize=14)
# plt.show()
# # # plt.savefig("/home/kia/catkin_ws/src/data_collection_human_test/data/pilot_plots/subject_001_baseline_acc.png")

# plt.figure()

for i in range(len(rotation_full_cont)):
    alignment = dtw(rotation_full_cont[5],rotation_full_cont[i], keep_internals=True, 
    step_pattern=rabinerJuangStepPattern(6, "c"))

    rebuilt_vec = np.zeros(len(alignment.index2))
    for index, val in enumerate(alignment.index2):
        rebuilt_vec[index] = rotation_full_cont[i][val]
    
    rotation_full_cont_aligned.append(rebuilt_vec)


fig, ax = plt.subplots(ncols=2, nrows=2, sharey='row', sharex='col', facecolor='whitesmoke')
for i in range(len(acceleration_full_aligned)):
    ax[0, 0].plot(np.arange(len(smooth(acceleration_full_aligned[i])))/60, smooth(acceleration_full_aligned[i]), linewidth=1.8)
for i in range(len(acceleration_full_cont_aligned)):
    ax[0, 1].plot(np.arange(len(smooth(acceleration_full_cont_aligned[i])))/60, smooth(acceleration_full_cont_aligned[i]), linewidth=1.8)
for i in range(len(rotation_full_aligned)):
    ax[1, 0].plot(np.arange(len(smooth(rotation_full_aligned[i])))/100, smooth(rotation_full_aligned[i]), linewidth=1.8)
for i in range(len(rotation_full_cont_aligned)):
    ax[1, 1].plot(np.arange(len(smooth(rotation_full_cont_aligned[i])))/80, smooth(rotation_full_cont_aligned[i]), linewidth=1.8)

ax[0, 0].spines['top'].set_visible(False)
ax[0, 0].spines['bottom'].set_visible(False)
ax[0, 1].spines['top'].set_visible(False)
ax[0, 1].spines['bottom'].set_visible(False)
ax[0, 0].spines['right'].set_visible(False)
ax[0, 1].spines['right'].set_visible(False)
ax[1, 0].spines['top'].set_visible(False)
ax[1, 1].spines['top'].set_visible(False)
ax[1, 0].spines['right'].set_visible(False)
ax[1, 1].spines['right'].set_visible(False)

ax[0, 0].yaxis.grid(alpha=0.5)
ax[0, 1].yaxis.grid(alpha=0.5)
ax[1, 0].yaxis.grid(alpha=0.5)
ax[1, 1].yaxis.grid(alpha=0.5)

ax[0, 0].tick_params(
    axis='x',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    bottom=False,      # ticks along the bottom edge are off
    top=False,         # ticks along the top edge are off
    labelbottom=False) # labels along the bottom edge are off
ax[0, 1].tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False) 

fig.text(0.5, 0.018, 'Time (s)', ha='center', size=16)
ax[0, 0].set_ylabel("acc ($m/s^{2}$)", fontsize=16, labelpad=16)
ax[1, 0].set_ylabel("rot (deg)", fontsize=16, labelpad=8)
ax[0, 0].tick_params(axis='both', labelsize=14)
ax[0, 0].tick_params(axis='y', pad=10)
ax[1, 0].tick_params(axis='y', pad=10)
ax[1, 0].tick_params(axis='both', labelsize=14)
ax[1, 1].tick_params(axis='both', labelsize=14)
ax[1, 1].set_xticks([0.0, 0.5, 1.0, 1.5, 2.0, 2.5])
ax[1, 1].xaxis.set_major_formatter(FormatStrFormatter('%.1f'))
ax[0, 0].set_title("Baseline", fontsize=16, pad=15, c='darkblue')
ax[0, 1].set_title("Controlled", fontsize=16, pad=15, c='black')

rect = plt.Rectangle(
    # (lower-left corner), width, height
    (0.11, 0.1), 0.39, 0.80, fill=False, color='blue', alpha=0.4, lw=1, 
    zorder=1000, transform=fig.transFigure, figure=fig, capstyle='round'
)
fig.patches.extend([rect])

rect1 = plt.Rectangle(
    # (lower-left corner), width, height
    (0.528, 0.1), 0.40, 0.80, fill=False, color='black', alpha=0.6, lw=1, 
    zorder=1000, transform=fig.transFigure, figure=fig, capstyle='round'
)
fig.patches.extend([rect1])

plt.subplots_adjust(bottom=0.19)
# plt.savefig('/home/kia/catkin_ws/src/data_collection_human_test/data/pilot_plots/result1.png')
plt.show()


# # plt.xticks([25*i for i in range(int(len(hand_imu)/25)+1)])
# # plt.grid()
# plt.xlabel("time step (60hz)", fontsize=16)
# plt.ylabel("acceleration (m/s^2)", fontsize=16, labelpad=-2)
# # plt.ylabel("rotatoin (deg)", fontsize=16, labelpad=-2)
# plt.title("Subject hand acceleration", fontsize=18)
# # plt.title("Object rotation", fontsize=18)
# plt.xticks(fontsize=14)
# plt.yticks(fontsize=14)
# plt.show()
# plt.savefig("/home/kia/catkin_ws/src/data_collection_human_test/data/pilot_plots/subject_004_cont_acc.png")
# print(max_normal_force)
# np.save("/home/kia/catkin_ws/src/data_collection_human_test/data/pilot_plots/subject_004_cont_max_grip.npy", max_normal_force)

#################################################################
# subject_004 baseline acc

    # if index == 0:
    #     start_index = 260
    #     end_index   = -1
    # elif index ==1:
    #     start_index = 285
    #     end_index   = -1
    # elif index ==2:
    #     start_index = 270
    #     end_index   = -10
    # elif index ==3:
    #     start_index = 255
    #     end_index   = -1
    # elif index ==4:
    #     start_index = 275
    #     end_index   = -1
    # elif index ==5:
    #     start_index = 275
    #     end_index   = -10
    # elif index ==6:
    #     start_index = 276
    #     end_index   = -1
    # elif index ==7:
    #     start_index = 270
    #     end_index   = -5
    # elif index ==8:
    #     start_index = 275
    #     end_index   = -5
    # elif index ==9:
    #     start_index = 280
    #     end_index   = -5


# subject_004 baseline rot
# easy to plot
# take axis 1 from the euler instead of 0

# subject_004 controlled acc

 # if index == 0:
    # continue
    # elif index ==1:
    #     start_index = 230
    #     end_index   = -1
    # elif index ==2:
    #     start_index = 225
    #     end_index   = -10
    # elif index ==3:
    #     start_index = 258
    #     end_index   = -1
    # elif index ==4:
    #     start_index = 285
    #     end_index   = -1
    # elif index ==5:
    #     start_index = 250
    #     end_index   = -10
    # elif index ==6:
    #     start_index = 253
    #     end_index   = -1
    # elif index ==7:
    #     start_index = 280
    #     end_index   = -3
    # elif index ==8:
    #     start_index = 255
    #     end_index   = -1
    # elif index ==9:
    #     start_index = 270
    #     end_index   = -5


# subject_004 controlled rot

 # if index == 0:
    # continue
    # elif index ==1:
    #     start_index = 226
    #     end_index   = -1
    # elif index ==2:
    #     start_index = 223
    #     end_index   = -1
    # elif index ==3:
    #     start_index = 232
    #     end_index   = -1
    # elif index ==4:
    #     start_index = 270
    #     end_index   = -1
    # elif index ==5:
    #     start_index = 225
    #     end_index   = -10
    # elif index ==6:
    #     start_index = 220
    #     end_index   = -1
    # elif index ==7:
    #     start_index = 250
    #     end_index   = -1
    # elif index ==8:
    #     start_index = 220
    #     end_index   = -1
    # elif index ==9:
    #     start_index = 225
    #     end_index   = -1

#################################################################


#################################################################
# subject_003 baseline acc

 # if index == 0:
    #     start_index = 55+280
    #     end_index   = 400
    # elif index ==1:
    #     start_index = 55+275
    #     end_index   = 400
    # elif index ==2:
    #     start_index = 55+250
    #     end_index   = 360
    # elif index ==3:
    #     start_index = 55+235
    #     end_index   = 338
    # elif index ==4:
    #     start_index = 55+250
    #     end_index   = -1
    # elif index ==5:
    #     start_index = 55+275
    #     end_index   = -5
    # elif index ==6:
    #     start_index = 55+248
    #     end_index   = 350
    # elif index ==7:
    #     start_index = 55+240
    #     end_index   = 350
    # elif index ==8:
    #     start_index = 55+225
    #     end_index   = 330
    # elif index ==9:
    #     start_index = 55+215
    #     end_index   = -5


# subject_003 baseline rot

 # if index == 0:
    #     start_index = 280
    #     end_index   = 400
    # elif index ==1:
    #     start_index = 275
    #     end_index   = 400
    # elif index ==2:
    #     start_index = 250
    #     end_index   = 360
    # elif index ==3:
    #     start_index = 235
    #     end_index   = 338
    # elif index ==4:
    #     start_index = 250
    #     end_index   = -1
    # elif index ==5:
    #     start_index = 275
    #     end_index   = -5
    # elif index ==6:
    #     start_index = 248
    #     end_index   = 350
    # elif index ==7:
    #     start_index = 240
    #     end_index   = 350
    # elif index ==8:
    #     start_index = 225
    #     end_index   = 330
    # elif index ==9:
    #     start_index = 215
    #     end_index   = -5


# subject_003 controlled acc

# if index == 0:
    #     start_index = 220
    #     end_index   = -1
    # elif index ==1:
    #     start_index = 290
    #     end_index   = -1
    # elif index ==2:
    #     start_index = 300
    #     end_index   = -1
    # elif index ==3:
    #     start_index = 285
    #     end_index   = -1
    # elif index ==4:
    #     start_index = 310
    #     end_index   = -1
    # elif index ==5:
    #     start_index = 255
    #     end_index   = -10
    # elif index ==6:
    #     start_index = 240
    #     end_index   = -5
    # elif index ==7:
    #     start_index = 245
    #     end_index   = -3
    # elif index ==8:
    #     start_index = 240
    #     end_index   = -5
    # elif index ==9:
    #     start_index = 225
    #     end_index   = -5


# subject_003 controlled rot

# if index == 0:
    #     start_index = 250
    #     end_index   = -1
    # elif index ==1:
    #     start_index = 240
    #     end_index   = -1
    # elif index ==2:
    #     start_index = 225
    #     end_index   = -1
    # elif index ==3:
    #     start_index = 215
    #     end_index   = -1
    # elif index ==4:
    #     start_index = 230
    #     end_index   = -1
    # elif index ==5:
    #     start_index = 245
    #     end_index   = -1
    # elif index ==6:
    #     start_index = 222
    #     end_index   = -1
    # elif index ==7:
    #     start_index = 225
    #     end_index   = -1
    # elif index ==8:
    #     start_index = 225
    #     end_index   = -1
    # elif index ==9:
    #     start_index = 225
    #     end_index   = -1

#################################################################


#################################################################
# subject_002 baseline acc

# if index == 0:
    #     start_index = 30 + 195
    #     end_index   = -5
    # elif index ==1:
    #     start_index = 30 + 200
    #     end_index   = -1
    # elif index ==2:
    #     start_index = 30 + 228
    #     end_index   = -1
    # elif index ==3:
    #     start_index = 30 + 210
    #     end_index   = -5
    # elif index ==4:
    #     start_index = 30 + 200
    #     end_index   = -1
    # elif index ==5:
    #     start_index = 30 + 200
    #     end_index   = 300
    # elif index ==6:
    #     start_index = 30 + 200
    #     end_index   = 300
    # elif index ==7:
    #     start_index = 30 + 200
    #     end_index   = -5
    # elif index ==8:
    #     start_index = 30 + 200
    #     end_index   = 300
    # elif index ==9:
    #     start_index = 30 + 180
    #     end_index   = 280


# subject_002 baseline rot

# if index == 0:
    #     start_index = 205
    #     end_index   = -5
    # elif index ==1:
    #     start_index = 200
    #     end_index   = -1
    # elif index ==2:
    #     start_index = 228
    #     end_index   = -1
    # elif index ==3:
    #     start_index = 210
    #     end_index   = -5
    # elif index ==4:
    #     start_index = 210
    #     end_index   = -1
    # elif index ==5:
    #     start_index = 200
    #     end_index   = 300
    # elif index ==6:
    #     start_index = 200
    #     end_index   = 300
    # elif index ==7:
    #     start_index = 200
    #     end_index   = -5
    # elif index ==8:
    #     start_index = 200
    #     end_index   = 300
    # elif index ==9:
    #     start_index = 180
    #     end_index   = 280


# subject_002 controlled acc

# if index == 0:
    #     start_index = 252
    #     end_index   = -1
    # elif index ==1:
    #     start_index = 250
    #     end_index   = -1
    # elif index ==2:
    #     start_index = 255
    #     end_index   = -1
    # elif index ==3:
    #     start_index = 252
    #     end_index   = -1
    # elif index ==4:
    #     start_index = 270
    #     end_index   = -1
    # elif index ==5:
    #     start_index = 260
    #     end_index   = -1
    # elif index ==6:
    #     start_index = 250
    #     end_index   = -1
    # elif index ==7:
    #     start_index = 250
    #     end_index   = -1
    # elif index ==8:
    #     start_index = 260
    #     end_index   = -1
    # elif index ==9:
    #     start_index = 200
    #     end_index   = -1

# subject_002 controlled rot

# if index == 0:
    #     start_index = 220
    #     end_index   = -1
    # elif index ==1:
    #     start_index = 205
    #     end_index   = -1
    # elif index ==2:
    #     start_index = 215
    #     end_index   = -1
    # elif index ==3:
    #     start_index = 215
    #     end_index   = -1
    # elif index ==4:
    #     start_index = 215
    #     end_index   = -1
    # elif index ==5:
    #     start_index = 190
    #     end_index   = -1
    # elif index ==6:
    #     start_index = 170
    #     end_index   = -1
    # elif index ==7:
    #     start_index = 200
    #     end_index   = -1
    # elif index ==8:
    #     start_index = 170
    #     end_index   = -1
    # elif index ==9:
    #     start_index = 180
    #     end_index   = -1

#################################################################

#################################################################
# subject_001 baseline acc

# if index == 0:
    #     start_index = 250
    #     end_index   = 300
        # continue
    # elif index ==1:
    #     start_index = 278
    #     end_index   = 342
    # elif index ==2:
    #     start_index = 275
    #     end_index   = -1
    # elif index ==3:
    #     start_index = 272
    #     end_index   = 310
    # elif index ==4:
    #     start_index = 275
    #     end_index   = 320
    # elif index ==5:
    #     start_index = 212
    #     end_index   = 279
    # elif index ==6:
    #     start_index = 212
    #     end_index   = 276
    # elif index ==7:
    #     start_index = 208
    #     end_index   = 271
    # elif index ==8:
    #     start_index = 206
    #     end_index   = 271
    # elif index ==9:
    #     start_index = 225
    #     end_index   = 290
        # continue


# subject_001 baseline rot
# all tasks: 204:304


# subject_001 controlled acc

# if index == 0:
    #     start_index = 230
    #     end_index   = 370
    # elif index ==1:
    #     start_index = 215
    #     end_index   = 336
    # elif index ==2:
    #     start_index = 240
    #     end_index   = 330
    # elif index ==3:
    #     start_index = 225
    #     end_index   = 371
    # elif index ==4:
    #     start_index = 230
    #     end_index   = 350
    # elif index ==5:
    #     start_index = 260
    #     end_index   = 370
    # elif index ==6:
    #     start_index = 248
    #     end_index   = -3
    # elif index ==7:
    #     start_index = 226
    #     end_index   = -3
    # elif index ==8:
    #     start_index = 258
    #     end_index   = -3
    # elif index ==9:
    #     start_index = 245
    #     end_index   = -1


# subject_001 controlled rot

# if index == 0:
    #     start_index = 195
    #     end_index   = 350
    # elif index ==1:
    #     start_index = 190
    #     end_index   = 350
    # elif index ==2:
    #     start_index = 200
    #     end_index   = 350
    # elif index ==3:
    #     start_index = 190
    #     end_index   = 370
    # elif index ==4:
    #     start_index = 195
    #     end_index   = -2
    # elif index ==5:
    #     start_index = 218
    #     end_index   = -1
    # elif index ==6:
    #     start_index = 200
    #     end_index   = -3
    # elif index ==7:
    #     start_index = 185
    #     end_index   = -1
    # elif index ==8:
    #     start_index = 225
    #     end_index   = -1
    # continue
    # elif index ==9:
    #     start_index = 225
    #     end_index   = -1
    # continue

#################################################################