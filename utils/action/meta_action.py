import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import action.gen3_move_cartesian as move
import action.gen3_move_gripper as gripper
import action.gen3_move_gripper_low as gripperl
import time
import action.utilities as utilities

from kortex_api.autogen.client_stubs.BaseClientRpc import BaseClient
from kortex_api.autogen.client_stubs.BaseCyclicClientRpc import BaseCyclicClient
from kortex_api.autogen.messages import Base_pb2, BaseCyclic_pb2, Common_pb2

Args = None

class ActionSequence:
    def __init__(self):
        """
        Use Execute function to transfer command to robot arm.
        Functions with the first letter being lowercase means it will take effect after "Execute".
        Functions with the first letter being uppercase means it will take effect immediately.
        """
        global Args
        if Args == None:
            Connect()

        self.action_list=[]
        #[type=0,x,y,z,tx,ty,tz]
        #[type=1,speed]
        #[type=2,time]
        #type=0 move; type=1 gripper; type=2 wait
        self.action_sequence_size = 0
        self.virtual_position,self.initial_gripper_position = Get_position()
        self.initial_position = self.virtual_position
        self.grasp_position = None


    # meta actions below:

    def move_to(self, target):
        """
        Intput: [x, y, z] or [x, y, z, theta_x, theta_y, theta_z]
        """
        self.action_list.append([0] + target)
        for i, p in enumerate(target):
            self.virtual_position[i] = p

    def move_gripper(self, position): # close: position = 100
        """
        close: position = 100, open: position = 0
        """
        # if speed < 0:
        #     self.grasp_position = self.virtual_position.copy()
        # elif speed > 0:
        #     self.grasp_position = None
        # if speed != 0:
        self.action_list.append([1, position])

    def wait(self, time):
        """
        Intput: seconds
        """
        self.action_list.append([2, time])
    
    # always used actions below: 

    def dump(self, degree, wait_time):
        """
        Intput: degree, and time stayed after dump
        """
        position = self.virtual_position.copy()
        position[4] += degree
        self.action_list.append([0] + position)

        self.wait(wait_time)

        position = self.virtual_position.copy()
        self.action_list.append([0] + position)
    
    def shake(self, strength):
        """
        Intput: distance of the shake action
        """
        position = self.virtual_position.copy()
        position[0] += strength/2
        position[1] += strength/2
        self.action_list.append([0] + position)
        position[0] -= strength
        self.action_list.append([0] + position)
        position[1] -= strength
        self.action_list.append([0] + position)
        position[0] += strength
        self.action_list.append([0] + position)
        position[1] += strength
        self.action_list.append([0] + position)
        position = self.virtual_position.copy()
        self.action_list.append([0] + position)
        
    def put_back(self):
        """
        Experimental!
        """
        if self.grasp_position != None:
            self.move_to(self.grasp_position)
            self.move_gripper(0.1)
        self.move_to(self.initial_position)
    
    # misc

    def Get_virtual_position(self):
        return self.virtual_position

    def Execute(self):
        print("Transfer to move_action ...")
        action_list_temp = []
        error = 0
        for action_data in self.action_list:
            type = action_data[0]
            action = action_data[1:]
            if type == 0:
                action_list_temp.append(action)
            elif type == 1:
                if action_list_temp:
                    error |= move.move_action(Args, action_list_temp, False)
                    action_list_temp = []
                error |= gripperl.gripper_action(Args, action[0])
            elif type == 2:
                if action_list_temp:
                    error |= move.move_action(Args, action_list_temp, False)
                    action_list_temp = []
                print("Waiting...")
                time.sleep(action[0])
            else:
                print("Action type is wrong!")
        if action_list_temp:
            error |= move.move_action(Args, action_list_temp, False)
        action_list_temp.clear()
        return error
    
    def Go_to_home_position(self):
        Go_to_home_position()
    
    def Get_position(self):
        return Get_position()
    
def Go_to_home_position():
    global Args
    if Args == None:
        Connect()
    move.move_action(Args, None, True)

def Get_position():
    global Args
    if Args == None:
        Connect()
    return move.get_feedback(Args), gripperl.get_feedback(Args)

def Connect():
    global Args
    if Args == None:
        Args = utilities.parseConnectionArguments()
        print(Args)