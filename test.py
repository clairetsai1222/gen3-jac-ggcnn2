import meta_action as ma

action = ma.ActionSequence()
action2 = ma.ActionSequence()
position = ma.get_position()
print(position)
position[2] += 10
action.move_gripper(-0.1)
action.move_to(position)
action.shake(-20)
action.wait(1)
action.dump(90,1)
action.put_back()
res = action.execute()
position = ma.get_position()
print(position)
print(res)