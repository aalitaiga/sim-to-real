import time

import math
from pypot.robot import from_remote
poppy_def = from_remote('flogo4.local', 4242)
poppy_att = from_remote('flogo2.local', 4242)

TEST_FOR = 10 # seconds

fpss = []
buff = []
start = time.time()
counter = 0
while True:
    pos_att = [m.present_position for m in poppy_att.motors]
    pos_def = [m.present_position for m in poppy_def.motors]
    buff.append ((pos_att, pos_def))
    current = time.time()
    if current - start >= 1:
        fps = len(buff) / (current-start)
        print ("{} FPS".format(fps))
        fpss.append(fps)
        buff = []
        start = time.time()
        counter += 1
        if counter == TEST_FOR:
            break

print ("AVG FPS",sum(fpss)/len(fpss))




