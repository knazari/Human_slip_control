# from vpython import *
import time
import math
import os
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FormatStrFormatter
# x = box(length=6, width=2, height=0.2)

# def sigmoid_time(x):
#   return -1 / (1 + math.exp(-(x-50)/14)) + 1

def sigmoid_time(x, mean, scale_suc, scale_fail):
    if x <= mean:
        x = -1 / (1 + math.exp(-(x-mean)/scale_suc)) + 1
    else:
        x = -1 / (1 + math.exp(-(x-mean)/scale_fail)) + 1
    return x
  
#   return -1 / (1 + math.exp(-(x-50)/14)) + 1

def sigmoid_rot(x, mean, scale_suc, scale_fail):
  x = -1 / (1 + math.exp(-(x-mean)/scale_suc)) + 1
  return x

def sigmoid_scaling(x, mean_, scale_suc):
    
    x = 1 / (1 + math.exp(-(x-mean_)/scale_suc)) 
    return x
    

mean_time = 60
scale_suc_time = 5
scale_fail_time = 20

mean_rot = 15
scale_suc_rot = 1.3
scale_fail_rot = 8

sigmoid_time_vec = np.vectorize(sigmoid_time)
x_vec = np.linspace(30, 150, 140)

ax = plt.axes()
ax.plot(x_vec, sigmoid_time_vec(x_vec[:], mean_time, scale_suc_time, scale_fail_time), linewidth=2, c='blue')
ax.hlines(sigmoid_time(30, mean_time, scale_suc_time, scale_fail_time), -20, 30, color='black', alpha=0.6, linewidth=0.6)
ax.hlines(sigmoid_time(mean_time, mean_time, scale_suc_time, scale_fail_time), -20, mean_time, color='black', alpha=0.6, linewidth=0.6)
ax.hlines(sigmoid_time(120, mean_time, scale_suc_time, scale_fail_time), -20, 120, color='black', alpha=0.6, linewidth=0.6)
ax.vlines(30, 0, sigmoid_time(30, mean_time, scale_suc_time, scale_fail_time), color='black', alpha=0.6, linewidth=0.6)
ax.vlines(mean_time, 0, sigmoid_time(mean_time, mean_time, scale_suc_time, scale_fail_time), color='black', alpha=0.6, linewidth=0.6)
ax.vlines(120, 0, sigmoid_time(120, mean_time, scale_suc_time, scale_fail_time), color='black', alpha=0.6, linewidth=0.6)
ax.set_xticks(np.arange(0, 110, 30))
ax.set_yticks([0, sigmoid_time(30, mean_time, scale_suc_time, scale_fail_time), sigmoid_time(mean_time, mean_time, scale_suc_time, scale_fail_time), sigmoid_time(120, mean_time, scale_suc_time, scale_fail_time), 1.0])
ax.tick_params(axis='both', which='major', labelsize=14)
ax.set_title("Time score", fontsize=22)
ax.set_xlabel("Time increase w.r.t baseline average (%)", fontsize=16, labelpad=8)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.figure()

sigmoid_rot_vec = np.vectorize(sigmoid_rot)
x_vec = np.linspace(10, 40, 30)

ax = plt.axes()
ax.plot(x_vec, sigmoid_rot_vec(x_vec[:], mean_rot, scale_suc_rot, scale_fail_rot), linewidth=2, c='blue')
ax.hlines(sigmoid_rot(10, mean_rot, scale_suc_rot, scale_fail_rot), 0, 10, color='black', alpha=0.6, linewidth=0.6)
ax.hlines(sigmoid_rot(mean_rot, mean_rot, scale_suc_rot, scale_fail_rot), 0, mean_rot, color='black', alpha=0.6, linewidth=0.6)
ax.hlines(sigmoid_rot(25, mean_rot, scale_suc_rot, scale_fail_rot), 0, 25, color='black', alpha=0.6, linewidth=0.6)
ax.vlines(10, 0, sigmoid_rot(10, mean_rot, scale_suc_rot, scale_fail_rot), color='black', alpha=0.6, linewidth=0.6)
ax.vlines(mean_rot, 0, sigmoid_rot(mean_rot, mean_rot, scale_suc_rot, scale_fail_rot), color='black', alpha=0.6, linewidth=0.6)
ax.vlines(25, 0, sigmoid_rot(25, mean_rot, scale_suc_rot, scale_fail_rot), color='black', alpha=0.6, linewidth=0.6)
ax.set_xticks([0, 10, 20, 25, 30])
ax.set_yticks([sigmoid_rot(10, mean_rot, scale_suc_rot, scale_fail_rot), sigmoid_rot(20, mean_rot, scale_suc_rot, scale_fail_rot), sigmoid_rot(25, mean_rot, scale_suc_rot, scale_fail_rot)])
ax.tick_params(axis='both', which='major', labelsize=14)
ax.set_title("Rotation score", fontsize=22)
ax.set_xlabel("Object rotation (deg)", fontsize=16, labelpad=8)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))


sigmoid_scaling_vec = np.vectorize(sigmoid_scaling)
s1_vec = np.linspace(0, 1, 100)
s2_vec = np.linspace(0, 1, 100)


def final_score(s1, s2):
  scaling_factor = sigmoid_scaling_vec(s2, 0.5, 0.1)
  s = 2 * scaling_factor * ((((s1 * 100) ** 1.8*s2) / 100) / 40 + 0.2)
  s_clipped = np.where(s < 1.0, s, 1.0)

  return s_clipped

func_value = final_score(s1_vec[:, None], s2_vec[None, :])

fig = plt.figure()
ax = plt.axes(projection='3d')
X, Y = np.meshgrid(s1_vec, s2_vec)
ax.contour3D(X, Y, func_value, 200, cmap='brg')
ax.set_xlabel('Time Score')
ax.set_ylabel('Rotation Score')
ax.set_zlabel('Combined Score');

# print(test_score(0.38, 0.93))

plt.show()

print(final_score(0, 1))
print(final_score(1, 0))
print(final_score(0.3, 0.5))


mean_time = 0.95
scale_suc_time = 0.3
scale_fail_time = 0.3

sigmoid_time_vec = np.vectorize(sigmoid_time)
x_vec = np.linspace(0, 2, 30)

ax = plt.axes()
ax.plot(x_vec, sigmoid_time_vec(x_vec[:], mean_time, scale_suc_time, scale_fail_time), linewidth=2, c='blue')
ax.set_xlabel("time value", fontsize=18)
ax.set_ylabel("time score", fontsize=18)
ax.set_title("Baseline time score sigmoid", fontsize=20)
plt.show()