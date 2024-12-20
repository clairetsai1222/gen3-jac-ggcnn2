import utils.action.meta_action as ma

action = ma.ActionSequence()
action2 = ma.ActionSequence()
position = ma.Get_position()
print(position)
position[2] += 10
action.move_gripper(-0.1)
action.move_to(position)
action.shake(2)
action.wait(1)
action.dump(90,1)
action.put_back()
res = action.Execute()
position = ma.Get_position()
print(position)
print(res)