import time

import zmq
import numpy as np
from poppy_helpers.randomizer import Randomizer

ROBOT1 = "flogo2.local"
PORT = 5757

context = zmq.Context()
socket = context.socket(zmq.PAIR)
print ("Connecting to server...")
socket.connect ("tcp://{}:{}".format(ROBOT1, PORT))
print ("Connected.")



observations = []
rand = Randomizer()

for i in range(1000):
    req = {"robot": {"get_pos_speed": {}}}
    socket.send_json(req)
    answer = socket.recv_json()
    observations.append(answer)

    if i % 100 == 0:
        req = {"robot": {"set_pos": {"positions":rand.random_sf(scaling=1.0)}}}
        socket.send_json(req)
        answer = socket.recv_json()


observations = np.array(observations)

print (observations.max(axis=0))
print (observations.min(axis=0))
