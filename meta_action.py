import gen3_move_cartesiancopy as move
import gen3_gripper_commandcopy as gripper
import time

class ActionSequence:
 
    def __init__(self):
        self.action_list=[]
        #[type=0,x,y,z,tx,ty,tz]
        #[type=1,speed]
        #[type=2,time]
        #type=0 move; type=1 gripper; type=2 wait
        self.action_sequence_size = 0
        self.current_position = move.get_feedback()
        self.initial_position = self.current_position
        self.grasp_position = None

    # meta actions below:

    def move_to(self, target):
        self.action_list.append([0] + target)
        for i, p in enumerate(target):
            self.current_position[i] = p

    def move_gripper(self, speed): # close: speed < 0
        if speed < 0:
            self.grasp_position = self.current_position
        if speed > 0:
            self.grasp_position = None
        if speed != 0:
            self.action_list.append([1, speed])

    def wait(self, time):
        self.action_list.append([2, time])
    
    # always used actions below: 

    def dump(self, degree, wait_time):
        position = self.current_position
        position[4] += degree
        self.action_list.append([0] + position)

        self.wait(wait_time)

        position = self.current_position
        self.action_list.append([0] + position)
    
    def shake(self, strength):
        position = self.current_position
        position[4] += strength
        self.action_list.append([0] + position)
        position[4] -= strength * 2
        self.action_list.append([0] + position)
        position = self.current_position
        self.action_list.append([0] + position)
        
    def put_back(self):
        if self.grasp_position != None:
            self.move_to(self.grasp_position)
            self.move_gripper(0.1)
        self.move_to(self.initial_position)
    
    # misc

    def get_current_position(self):
        return self.current_position

    def execute(self):
        print("Transfer to move_action ...")
        action_list_temp = []
        error = 0
        for action_data in self.action_list:
            type = action_data[0]
            action = action_data[1:]
            if action_data == 0:
                action_list_temp.append(action)
            elif action_data == 1:
                if action_list_temp:
                    error |= move.move_action(action_list_temp)
                    action_list_temp = []
                error |= gripper.gripper_action(action[0])
            elif action_data == 2:
                if action_list_temp:
                    error |= move.move_action(action_list_temp)
                    action_list_temp = []
                time.sleep(action[0])
            else:
                print("Action type is wrong!")
        if action_list_temp:
            error |= move.move_action(action_list_temp)
        action_list_temp.clear()
        return error