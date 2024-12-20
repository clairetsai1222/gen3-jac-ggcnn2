#! /usr/bin/env python3

###
# KINOVA (R) KORTEX (TM)
#
# Copyright (c) 2019 Kinova inc. All rights reserved.
#
# This software may be modified and distributed under the
# terms of the BSD 3-Clause license.
#
# Refer to the LICENSE file for details.
#
###

import sys
import os
import time

from kortex_api.autogen.client_stubs.BaseClientRpc import BaseClient
from kortex_api.autogen.messages import Base_pb2

# Import the utilities helper module
import argparse
#sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import utilities

class GripperCommandExample:
    def __init__(self, router, proportional_gain = 2.0):

        self.proportional_gain = proportional_gain
        self.router = router

        # Create base client using TCP router
        self.base = BaseClient(self.router)

    def ExampleSendGripperCommands(self, speed, force_when_stop=1):

        # Create the GripperCommand we will send
        gripper_command = Base_pb2.GripperCommand()
        finger = gripper_command.gripper.finger.add()
        gripper_request = Base_pb2.GripperRequest()

        # Close the gripper with position increments
        #print("Performing gripper test in position...")
        #gripper_command.mode = Base_pb2.GRIPPER_POSITION
        #position = 0.00
        #finger.finger_identifier = 1
        #while position < 1.0:
        #    finger.value = position
        #    print("Going to position {:0.2f}...".format(finger.value))
        #    self.base.SendGripperCommand(gripper_command)
        #    position += 0.1
        #    time.sleep(1)

        # Set speed to open gripper
        #print ("Opening gripper using speed command...")
        #gripper_command.mode = Base_pb2.GRIPPER_SPEED
        #finger.value = 0.1
        #self.base.SendGripperCommand(gripper_command)

        # Wait for reported position to be opened
        #gripper_request.mode = Base_pb2.GRIPPER_POSITION
        #while True:
        #    gripper_measure = self.base.GetMeasuredGripperMovement(gripper_request)
        #    if len (gripper_measure.finger):
        #        print("Current position is : {0}".format(gripper_measure.finger[0].value))
        #        if gripper_measure.finger[0].value < 0.01:
        #            break
        #    else: # Else, no finger present in answer, end loop
        #        break

        # Set speed to close gripper
        print ("Moving gripper using speed command...")
        gripper_command.mode = Base_pb2.GRIPPER_SPEED
        finger.value = speed
        self.base.SendGripperCommand(gripper_command)

        # Wait for reported force
        gripper_request.mode = Base_pb2.GRIPPER_FORCE
        while True:
            gripper_measure = self.base.GetMeasuredGripperMovement(gripper_request)
            if len (gripper_measure.finger):
                print("Current force is : {0}".format(gripper_measure.finger[0].value))
                if gripper_measure.finger[0].value > force_when_stop:
                    # stop
                    finger.value = 0
                    self.base.SendGripperCommand(gripper_command)
                    break
            else: # Else, no finger present in answer, end loop
                return False
        # Wait for reported speed to be 0
        gripper_request.mode = Base_pb2.GRIPPER_SPEED
        while True:
            gripper_measure = self.base.GetMeasuredGripperMovement(gripper_request)
            if len (gripper_measure.finger):
                print("Current speed is : {0}".format(gripper_measure.finger[0].value))
                if gripper_measure.finger[0].value == 0.0:
                    return True
            else: # Else, no finger present in answer, end loop
                return False

def gripper_action(args, speed):

    # # Parse arguments
    # parser = argparse.ArgumentParser()
    # args = utilities.parseConnectionArguments(parser)

    # # Create connection to the device and get the router
    with utilities.DeviceConnection.createTcpConnection(args) as router:
        success = True
        example = GripperCommandExample(router)
        success &= example.ExampleSendGripperCommands(speed)
        return 0 if success else 1

if __name__ == "__main__":
    exit(gripper_action(0.1))