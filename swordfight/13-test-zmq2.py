import time

import zmq

ROBOT1 = "poppysou.local"
PORT = 5757

context = zmq.Context()
socket = context.socket(zmq.PAIR)
print ("Connecting to server...")
socket.connect ("tcp://{}:{}".format(ROBOT1, PORT))
print ("Connected.")




# req = {"robot": {"get_register_value": {"motor": "m2", "register": "present_load"}}}
# socket.send_json(req)
# answer = socket.recv_json()
# print(answer)

req = {"robot": {"get_motor_registers_list": {"motor": "m2"}}}
socket.send_json(req)
answer = socket.recv_json()
print(answer)

#     req = {"robot": {"get_pos_speed": {}}}
#     socket.send_json(req)
#     answer = socket.recv_json()
#     # print(answer)

req = {"robot": {"set_pos": {"positions":[45, 0, 0, 0, 0, 0]}}}
socket.send_json(req)
answer = socket.recv_json()
print(answer)

time.sleep(2)

#
req = {"robot": {"set_register_value": {"motor": "m1", "register": "goal_position", "value": "45"}}}
socket.send_json(req)
answer = socket.recv_json()
print(answer)

for i in range(6):
    req = {"robot": {"set_register_value": {"motor": "m{}".format(i+1), "register": "compliant", "value": "False"}}}
    socket.send_json(req)
    answer = socket.recv_json()
    print(answer)

req = {"robot": {"set_register_value": {"motor": "m2", "register": "led", "value": "cyan"}}}
socket.send_json(req)
answer = socket.recv_json()
print(answer)




