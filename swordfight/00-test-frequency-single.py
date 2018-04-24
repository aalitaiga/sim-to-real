import time

import math
from pypot.robot import from_remote
poppy = from_remote('flogo2.local', 4242)

poppy.dance.start()

TEST_FOR = 10 # seconds

fpss = []
buff = []
start = time.time()
counter = 0
while True:
    pos_att = [m.present_position for m in poppy.motors]
    buff.append (pos_att)
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
poppy.dance.stop()



