import glob
import time
import numpy as np
import pandas as pd
from scipy import integrate
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
from dtw import *
import rospy
from std_msgs.msg import Float64MultiArray

def smooth(x,window_len=11,window='hanning'):
    s = np.r_[x[window_len-1:0:-1],x,x[-2:-window_len-1:-1]]
    if window == 'flat': #moving average
        w=np.ones(window_len,'d')
    else:
        w=eval('np.'+window+'(window_len)')
    y=np.convolve(w/w.sum(),s,mode='valid')
    
    return y[int((window_len/2-1)):-int((window_len/2))]



#! /usr/bin/env python3
from time import *
import time
import rospy
import numpy as np
import math
import serial
from std_msgs.msg import Float64MultiArray
from geometry_msgs.msg import PoseStamped

ad1 = serial.Serial('/dev/ttyUSB2',115200)
sleep(1)

rospy.init_node('IMU_data', anonymous=True)
hand_imu_pub = rospy.Publisher('/hand_imu_rot', Float64MultiArray, queue_size=1)
hand_imu_pose_pub = rospy.Publisher('/hand_imu_rot_pose', PoseStamped, queue_size=1)

rate = rospy.Rate(1000)


while not rospy.is_shutdown():
    try:

        dataPacket1 = ad1.readline()

        dataPacket1 = str(dataPacket1,'utf-8')

        splitPacket1 = dataPacket1.split(",")

        object_imu_rot  = R.from_quat([float(splitPacket1[1]), float(splitPacket1[2]), float(splitPacket1[3]), float(splitPacket1[0])]).as_euler('xyx', degrees=True)
        hand_imu_msg = Float64MultiArray()
        hand_imu_msg.data = np.array(object_imu_rot)
        hand_imu_pub.publish(hand_imu_msg)

        object_pose = PoseStamped()
        object_pose.header.frame_id = "map"
        object_pose.pose.position.x = 0.0
        object_pose.pose.position.y = 0.0
        object_pose.pose.position.z = 0.0
        object_pose.pose.orientation.x = float(splitPacket1[1])
        object_pose.pose.orientation.y = float(splitPacket1[2])
        object_pose.pose.orientation.z = float(splitPacket1[3])
        object_pose.pose.orientation.w = float(splitPacket1[0])
        hand_imu_pose_pub.publish(object_pose)

        rate.sleep()

    except:
        pass
