import meta_action as ma

action = ma.ActionSequence()
position = action.get_current_position()
position[2] += 10
action.move_to(position)
action.move_gripper(0.1)
action.wait(2)
action.shake(10)
action.wait(1)
action.dump(2)
action.put_back()
res = action.execute()
print(res)