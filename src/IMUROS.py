#! /usr/bin/env python3
from time import *
import time
import rospy
import numpy as np
import math
import serial
from std_msgs.msg import Float64MultiArray

ad1 = serial.Serial('/dev/ttyUSB1',115200)
ad2 = serial.Serial('/dev/ttyUSB2',115200)
sleep(1)

rospy.init_node('IMU_data', anonymous=True)
hand_imu_pub = rospy.Publisher('/hand_imu', Float64MultiArray, queue_size=1)
object_imu_pub = rospy.Publisher('/object_imu', Float64MultiArray, queue_size=1)

rate = rospy.Rate(1000)

hand_v_x_0 = 0
hand_v_y_0 = 0
hand_v_z_0 = 0
object_v_x_0 = 0
object_v_y_0 = 0
object_v_z_0 = 0


while not rospy.is_shutdown():
    try:
        while (ad1.inWaiting()==0) or (ad2.inWaiting()==0):
            pass
        
        t1 = time.time()

        dataPacket1 = ad1.readline()
        dataPacket2 = ad2.readline()

        dataPacket1 = str(dataPacket1,'utf-8')
        dataPacket2 = str(dataPacket2,'utf-8')

        splitPacket1 = dataPacket1.split(",")
        splitPacket2 = dataPacket2.split(",")


        hand_q0 = float(splitPacket1[0])
        hand_q1 = float(splitPacket1[1])
        hand_q2 = float(splitPacket1[2])
        hand_q3 = float(splitPacket1[3])

        object_q0 = float(splitPacket2[0])
        object_q1 = float(splitPacket2[1])
        object_q2 = float(splitPacket2[2])
        object_q3 = float(splitPacket2[3])

        # hand_roll  = -math.atan2(2*(hand_q0*hand_q1+hand_q2*hand_q3),1-2*(hand_q1*hand_q1+hand_q2*hand_q2))
        # hand_pitch =  math.asin(2*(hand_q0*hand_q2-hand_q3*hand_q1))
        # hand_yaw   = -math.atan2(2*(hand_q0*hand_q3+hand_q1*hand_q2),1-2*(hand_q2*hand_q2+hand_q3*hand_q3))-np.pi/2

        # object_roll  = -math.atan2(2*(object_q0*object_q1+object_q2*object_q3),1-2*(object_q1*object_q1+object_q2*object_q2))
        # object_pitch =  math.asin(2*(object_q0*object_q2-object_q3*object_q1))
        # object_yaw   = -math.atan2(2*(object_q0*object_q3+object_q1*object_q2),1-2*(object_q2*object_q2+object_q3*object_q3))-np.pi/2

        # hand_roll  = hand_roll*180/np.pi
        # hand_pitch = hand_pitch*180/np.pi
        # hand_yaw   = hand_yaw*180/np.pi

        # object_roll  = object_roll*180/np.pi
        # object_pitch = object_pitch*180/np.pi
        # object_yaw   = object_yaw*180/np.pi


        hand_acc_x = float(splitPacket1[4])
        hand_acc_y = float(splitPacket1[5])
        hand_acc_z = float(splitPacket1[6])

        object_acc_x = float(splitPacket2[4])
        object_acc_y = float(splitPacket2[5])
        object_acc_z = float(splitPacket2[6])

        hand_imu_msg = Float64MultiArray()
        hand_imu_msg.data = np.array([hand_q0, hand_q1, hand_q2, hand_q3, hand_acc_x, hand_acc_y, hand_acc_z])
        hand_imu_pub.publish(hand_imu_msg)

        object_imu_msg = Float64MultiArray()
        object_imu_msg.data = np.array([object_q0, object_q1, object_q2, object_q3, object_acc_x, object_acc_y, object_acc_z])
        object_imu_pub.publish(object_imu_msg)

        rate.sleep()

    except:
        pass