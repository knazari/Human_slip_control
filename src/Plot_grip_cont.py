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
# 
    y=np.convolve(w/w.sum(),s,mode='valid')
    #return y
    return y[int((window_len/2-1)):-int((window_len/2))]

plt.rcParams["figure.figsize"] = (7,4)

user_path = "/home/kia/catkin_ws/src/data_collection_human_test/data/real_test2/subject_003/baseline"
files = sorted(glob.glob(user_path + "/*"))
normal_force_cont = []

for index, file in enumerate(files):

    meta_data = pd.read_csv(file + '/meta_data.csv')
    start_index = meta_data['start_index'][0] - 20
    end_index   = meta_data['end_index'][0] + 30

    xela_data = pd.read_csv(file + '/xela.csv')[['txl1_z', 'txl2_z', 'txl3_z', 'txl4_z', 'txl5_z', 'txl6_z',
                                    'txl7_z', 'txl8_z', 'txl9_z', 'txl10_z', 'txl11_z', 'txl12_z', 'txl13_z',
                                    'txl14_z', 'txl15_z', 'txl16_z']]
    
    xela_data = np.array(xela_data)
    xela_data = np.sum(xela_data, axis=1)
    plt.plot(xela_data, c='blue', linewidth=2)



user_path = "/home/kia/catkin_ws/src/data_collection_human_test/data/real_test2/subject_003/controlled"
files = sorted(glob.glob(user_path + "/*"))
normal_force_cont = []

for index, file in enumerate(files):

    meta_data = pd.read_csv(file + '/meta_data.csv')
    start_index = meta_data['start_index'][0] - 20
    end_index   = meta_data['end_index'][0] + 30

    xela_data = pd.read_csv(file + '/xela.csv')[['txl1_z', 'txl2_z', 'txl3_z', 'txl4_z', 'txl5_z', 'txl6_z',
                                    'txl7_z', 'txl8_z', 'txl9_z', 'txl10_z', 'txl11_z', 'txl12_z', 'txl13_z',
                                    'txl14_z', 'txl15_z', 'txl16_z']]
    
    xela_data = np.array(xela_data)
    xela_data = np.sum(xela_data, axis=1)
    plt.plot(xela_data, c='red', linewidth=2)

cmap = plt.cm.coolwarm
from matplotlib.lines import Line2D

custom_lines = [Line2D([0], [0], color=cmap(0.), lw=4),
                Line2D([0], [0], color=cmap(1.), lw=4)]

plt.legend(custom_lines, ['Baseline', 'Controlled'], fontsize=14, loc='upper left')
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.title("Sunject 3", fontsize=16)
plt.xlabel("time step", fontsize=16)
plt.ylabel("sum of normal forces", fontsize=20)
plt.show()
