#!/usr/bin/env python
import rospy
from std_msgs.msg import String
from std_msgs.msg import Float64MultiArray , Float64

xela1_pub = rospy.Publisher('/xela_sum_data', Float64, queue_size = 1)

def callback(data):
    # print(sum(data.data[0:16]))
    xela_msg = Float64()
    xela_msg.data = sum(data.data[40:41])
    xela1_pub.publish(xela_msg)
    
def listener():

    rospy.init_node('listener', anonymous=True)

    

    rospy.Subscriber('/xela1_data', Float64MultiArray, callback)

    # spin() simply keeps python from exiting until this node is stopped
    rospy.spin()

if __name__ == '__main__':
    listener()