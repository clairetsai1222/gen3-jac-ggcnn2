#! /usr/bin/env python3

###
# KINOVA (R) KORTEX (TM)
#
# Copyright (c) 2018 Kinova inc. All rights reserved.
#
# This software may be modified and distributed
# under the terms of the BSD 3-Clause license.
#
# Refer to the LICENSE file for details.
#
###

import sys
import os
import time
import threading

from collections.abc import Iterable
from kortex_api.autogen.client_stubs.BaseClientRpc import BaseClient
from kortex_api.autogen.client_stubs.BaseCyclicClientRpc import BaseCyclicClient

from kortex_api.autogen.messages import Base_pb2, BaseCyclic_pb2, Common_pb2

# Maximum allowed waiting time during actions (in seconds)
TIMEOUT_DURATION = 20

def check_for_sequence_end_or_abort(e):
    """Return a closure checking for END or ABORT notifications on a sequence

    Arguments:
    e -- event to signal when the action is completed
        (will be set when an END or ABORT occurs)
    """

    def check(notification, e = e):
        event_id = notification.event_identifier
        task_id = notification.task_index
        if event_id == Base_pb2.SEQUENCE_TASK_COMPLETED:
            print("Sequence task {} completed".format(task_id))
        elif event_id == Base_pb2.SEQUENCE_ABORTED:
            print("Sequence aborted with error {}:{}"\
                .format(\
                    notification.abort_details,\
                    Base_pb2.SubErrorCodes.Name(notification.abort_details)))
            e.set()
        elif event_id == Base_pb2.SEQUENCE_COMPLETED:
            print("Sequence completed.")
            e.set()
    return check

# Create closure to set an event after an END or an ABORT 
# 这个函数返回一个闭包，用于检查动作是否结束（END）或中止（ABORT），并在检测到这些事件时设置一个线程事件e
def check_for_end_or_abort(e):
    """Return a closure checking for END or ABORT notifications

    Arguments:
    e -- event to signal when the action is completed
        (will be set when an END or ABORT occurs)
    """
    def check(notification, e = e):
        print("EVENT : " + \
              Base_pb2.ActionEvent.Name(notification.action_event))
        if notification.action_event == Base_pb2.ACTION_END \
        or notification.action_event == Base_pb2.ACTION_ABORT:
            e.set()
    return check
 
def example_move_to_home_position(base):
    # Make sure the arm is in Single Level Servoing mode
    base_servo_mode = Base_pb2.ServoingModeInformation()
    base_servo_mode.servoing_mode = Base_pb2.SINGLE_LEVEL_SERVOING
    base.SetServoingMode(base_servo_mode)
    
    # Move arm to ready position
    print("Moving the arm to a safe position")
    action_type = Base_pb2.RequestedActionType()
    action_type.action_type = Base_pb2.REACH_JOINT_ANGLES
    action_list = base.ReadAllActions(action_type)
    action_handle = None
    for action in action_list.action_list:
        if action.name == "Home":
            action_handle = action.handle

    if action_handle == None:
        print("Can't reach safe position. Exiting")
        return False

    e = threading.Event()
    notification_handle = base.OnNotificationActionTopic(
        check_for_end_or_abort(e),
        Base_pb2.NotificationOptions()
    )

    base.ExecuteActionFromReference(action_handle)
    finished = e.wait(TIMEOUT_DURATION)
    base.Unsubscribe(notification_handle)

    if finished:
        print("Safe position reached")
    else:
        print("Timeout on action notification wait")
    return finished


def example_cartesian_action_movement_sequence(base, base_cyclic, action_list):
    '''
    Arg:
    base: an instance of BaseClient 基础客户端
    base_cyclic: an instance of BaseCyclicClient 循环基础客户端
    '''
    # 按顺序调用action_list内部的动作
    # 每一个 action in action_list 的前三位为point，后三位为angle
    print("Creating Sequence")
    action_sequence = Base_pb2.Sequence()
    action_sequence.name = "Example sequence"
    task = []
    
    print("Appending Actions to Sequence")
    if isinstance(action_sequence[0], Iterable):
        for i, target in enumerate(action_sequence):
            action = create_cartesian_action(base, base_cyclic, target)
            task.append(action_sequence.tasks.add())
            task[i].group_identifier = i
            task[i].action.CopyFrom(action)
    else:
        action = create_cartesian_action(base, base_cyclic, action_sequence)
        task.append(action_sequence.tasks.add())
        task[i].group_identifier = i
        task[i].action.CopyFrom(action)


    e = threading.Event()
    notification_handle = base.OnNotificationSequenceActionTopic(
        check_for_sequence_end_or_abort(e),
        Base_pb2.NotificationOptions()
    )
    print("Creating sequence on device and executing it")
    handle_sequence = base.CreateSequence(action_sequence)
    base.PlaySequence(handle_sequence)

    print("Waiting for movement to finish ...")
    finished = e.wait(TIMEOUT_DURATION)
    base.Unsubscribe(notification_handle)
    task.clear()

    if finished:
        print("Cartesian movement completed")
    else:
        print("Timeout on action notification wait")
    return finished

def create_cartesian_action(base, base_cyclic, target):
    print("Starting Cartesian action movement ...")
    action = Base_pb2.Action()
    action.name = "Example Cartesian action movement"
    action.application_data = "" 

    feedback = base_cyclic.RefreshFeedback() # 从base_cyclic客户端获取当前的反馈信息

    cartesian_pose = action.reach_pose.target_pose

    cartesian_pose.x = (target[0]+3)     *0.01     # (meters) 
    cartesian_pose.y = (target[1]+20)     *0.01   # (meters)
    cartesian_pose.z = target[2]     *0.01   # (meters)
    
    if len(target) == 6:
        cartesian_pose.theta_x = target[3] # (degrees)夹爪角度：+往下；-往上
        cartesian_pose.theta_y = target[4] # (degrees)夹爪角度：+逆时针；-顺时针
        cartesian_pose.theta_z = target[5] # (degrees)夹爪角度：+左转；-右转
    else:
        cartesian_pose.theta_x = feedback.base.tool_pose_theta_x # (degrees)夹爪角度：+往下；-往上
        cartesian_pose.theta_y = feedback.base.tool_pose_theta_y # (degrees)夹爪角度：+逆时针；-顺时针
        cartesian_pose.theta_z = feedback.base.tool_pose_theta_z # (degrees)夹爪角度：+左转；-右转

    # cartesian_pose.x = feedback.base.tool_pose_x 
    # cartesian_pose.y = feedback.base.tool_pose_y 
    # cartesian_pose.z = feedback.base.tool_pose_z 
    # cartesian_pose.theta_x = feedback.base.tool_pose_theta_x # (degrees)夹爪角度：+往下；-往上
    # cartesian_pose.theta_y = feedback.base.tool_pose_theta_y # (degrees)夹爪角度：+逆时针；-顺时针
    # cartesian_pose.theta_z = feedback.base.tool_pose_theta_z # (degrees)夹爪角度：+左转；-右转

    return action


def move_action(action_list):
    """
    Input: action_list
    [x, y, z] or [x, y, z, theta_x, theta_y, theta_z]
    or a list composed of the first two, and the actions will be executed in order.
    """
    # Import the utilities helper module
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    import utilities

    # Parse arguments
    args = utilities.parseConnectionArguments()
    
    # Create connection to the device and get the router
    with utilities.DeviceConnection.createTcpConnection(args) as router:

        # Create required services
        base = BaseClient(router)
        base_cyclic = BaseCyclicClient(router)

        # Example core
        success = True

        success &= example_move_to_home_position(base)
        success &= example_cartesian_action_movement_sequence(base, base_cyclic, action_list)

        # You can also refer to the 110-Waypoints examples if you want to execute
        # a trajectory defined by a series of waypoints in joint space or in Cartesian space

        return 0 if success else 1

if __name__ == "__main__":
    exit(move_action([80.37678837102317, -25.803701662536618, 4.438464752449033]))
