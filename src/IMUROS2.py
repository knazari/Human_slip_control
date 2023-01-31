#! /usr/bin/env python3
from time import *
import time
import rospy
import numpy as np
import math
import serial
from std_msgs.msg import Float64MultiArray


ad2 = serial.Serial('/dev/ttyUSB2',115200)
sleep(1)

rospy.init_node('IMU_data2', anonymous=True)
object_imu_pub = rospy.Publisher('/object_imu', Float64MultiArray, queue_size=1)

rate = rospy.Rate(1000)

while not rospy.is_shutdown():
    try:

        dataPacket2 = ad2.readline()

        dataPacket2 = str(dataPacket2,'utf-8')

        splitPacket2 = dataPacket2.split(",")

        object_imu_msg = Float64MultiArray()
        object_imu_msg.data = np.array([float(splitPacket2[0]), float(splitPacket2[1]), float(splitPacket2[2]), float(splitPacket2[3]),\
                                             float(splitPacket2[4]), float(splitPacket2[5]), float(splitPacket2[6])])
        object_imu_pub.publish(object_imu_msg)

        rate.sleep()

    except:
        pass