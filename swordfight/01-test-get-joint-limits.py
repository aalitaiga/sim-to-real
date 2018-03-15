from pypot.robot import from_remote

poppy_def = from_remote('flogo4.local', 4242)
for i in range(6):
    print (i, poppy_def.motors[0].lower_limit, poppy_def.motors[0].upper_limit)

print ("===")

poppy_att = from_remote('flogo2.local', 4242)
for i in range(6):
    print (i, poppy_att.motors[0].lower_limit, poppy_att.motors[0].upper_limit)

print ("===")

for item in dir(poppy_att.motors[0]):
    print(item)



