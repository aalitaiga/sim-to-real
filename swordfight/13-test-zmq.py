import time

import pypot
import zmq

ROBOT1 = "flogo4.local"
PORT = 5757

context = zmq.Context()
socket = context.socket(zmq.PAIR)
print ("Connecting to server...")
socket.connect ("tcp://{}:{}".format(ROBOT1, PORT))
print ("Connected.")




req = {"robot": {"get_register_value": {"motor": "m2", "register": "present_load"}}}
socket.send_json(req)
answer = socket.recv_json()
print(answer)

req = {"robot": {"get_motor_registers_list": {"motor": "m2"}}}
socket.send_json(req)
answer = socket.recv_json()
print(answer)

tests = 1000

start = time.time()
for i in range(tests):

    req = {"robot": {"get_pos_speed": {}}}
    socket.send_json(req)
    answer = socket.recv_json()
    # print(answer)

    req = {"robot": {"set_pos": {"pos":[0, 0, 0, 0, 0, 0]}}}
    socket.send_json(req)
    answer = socket.recv_json()
    # print(answer)

end = time.time()

print ("{} Hz".format(tests/(end-start)))

req = {"robot": {"set_register_value": {"motor": "m1", "register": "goal_position", "value": "20"}}}
socket.send_json(req)
answer = socket.recv_json()
print(answer)