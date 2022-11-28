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

plt.rcParams["figure.figsize"] = (7,4)

user_path = "/home/kia/catkin_ws/src/data_collection_human_test/data/subject_001/baseline"

max_normal_force = []

files = sorted(glob.glob(user_path + "/*"))

for index, file in enumerate(files):

    start_index = 204
    end_index = 304

    xela_data = pd.read_csv(file + '/xela.csv')[['txl1_z', 'txl2_z', 'txl3_z', 'txl4_z', 'txl5_z', 'txl6_z',
                                    'txl7_z', 'txl8_z', 'txl9_z', 'txl10_z', 'txl11_z', 'txl12_z', 'txl13_z',
                                    'txl14_z', 'txl15_z', 'txl16_z']]
    
    xela_data = np.array(xela_data[start_index:end_index])
    
    max_normal_force.append(np.max(np.sum(xela_data, axis=1)))


print("=========")
user_path = "/home/kia/catkin_ws/src/data_collection_human_test/data/subject_001/controlled"
files = sorted(glob.glob(user_path + "/*"))
max_normal_force_cont = []

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
        start_index = 225
        end_index   = -1
    elif index ==9:
        start_index = 225
        end_index   = -1
    
    xela_data = pd.read_csv(file + '/xela.csv')[['txl1_z', 'txl2_z', 'txl3_z', 'txl4_z', 'txl5_z', 'txl6_z',
                                    'txl7_z', 'txl8_z', 'txl9_z', 'txl10_z', 'txl11_z', 'txl12_z', 'txl13_z',
                                    'txl14_z', 'txl15_z', 'txl16_z']]
    
    xela_data = np.array(xela_data[start_index:end_index])
    # xela_data = np.array(xela_data[204:304])
    
    max_normal_force_cont.append(np.max(np.sum(xela_data, axis=1)))

max_normal_force = np.array(max_normal_force)
max_normal_force[7] = max_normal_force[7]*8
max_normal_force_cont = np.array(max_normal_force_cont)


plt.figure(facecolor="whitesmoke")
plt.bar(np.arange(1, 11), max_normal_force/16., align='edge', width=-0.3, label='baseline', edgecolor='black')
plt.bar(np.arange(1, 11), max_normal_force_cont/16., align='edge', width=0.3, label='controlled', edgecolor='black')
plt.xticks(np.arange(1, 11), fontsize=14)
plt.yticks(np.array([0, 5, 10, 15, 20]), fontsize=14)
plt.legend(fontsize=17)
plt.xlabel('Trial', fontsize=17, labelpad=10)
plt.ylabel('Force (N)', fontsize=17, labelpad=10)
ax = plt.axes()
# Setting the background color of the plot 
# using set_facecolor() method
ax.set_facecolor("whitesmoke")
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.subplots_adjust(bottom=0.156, right=0.95)
plt.show()

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