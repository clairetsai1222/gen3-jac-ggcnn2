import gen3_move_cartesiancopy as move
import gen3_gripper_commandcopy as gripper
import time
import utilities

from kortex_api.autogen.client_stubs.BaseClientRpc import BaseClient
from kortex_api.autogen.client_stubs.BaseCyclicClientRpc import BaseCyclicClient
from kortex_api.autogen.messages import Base_pb2, BaseCyclic_pb2, Common_pb2

Args = None

class ActionSequence:
    def __init__(self):
        """
        Use execute function to transfer command to robot arm.
        """
        global Router, Base, Base_cyclic
        if Args == None:
            connect()

        self.action_list=[]
        #[type=0,x,y,z,tx,ty,tz]
        #[type=1,speed]
        #[type=2,time]
        #type=0 move; type=1 gripper; type=2 wait
        self.action_sequence_size = 0
        self.virtual_position = get_position()
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

    def move_gripper(self, speed): # close: speed < 0
        """
        Intput: speed<0: close; speed>0 open
        """
        if speed < 0:
            self.grasp_position = self.virtual_position.copy()
        if speed > 0:
            self.grasp_position = None
        if speed != 0:
            self.action_list.append([1, speed])

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
        position = self.virtual_position
        position[4] += degree
        self.action_list.append([0] + position)

        self.wait(wait_time)

        position[4] -= degree
        self.action_list.append([0] + position)
    
    def shake(self, strength):
        """
        Intput: degree of the shake action
        """
        position = self.virtual_position
        position[4] += strength
        self.action_list.append([0] + position)
        position[4] -= strength * 2
        self.action_list.append([0] + position)
        position[4] += strength
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

    def get_virtual_position(self):
        return self.virtual_position

    def execute(self):
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
                error |= gripper.gripper_action(Args, action[0])
            elif type == 2:
                if action_list_temp:
                    error |= move.move_action(Args, action_list_temp, False)
                    action_list_temp = []
                time.sleep(action[0])
            else:
                print("Action type is wrong!")
        if action_list_temp:
            error |= move.move_action(Args, action_list_temp, False)
        action_list_temp.clear()
        return error
    
def get_position():
    if Args == None:
        connect()
    return move.get_feedback(Args)

def connect():
    global Args
    Args = utilities.parseConnectionArguments()