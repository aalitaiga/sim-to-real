import queue
import threading
import tkinter as tk
import time
import numpy as np
from pytorch_a2c_ppo_acktr.inference import Inference

from poppy_helpers.normalizer import Normalizer
from poppy_helpers.randomizer import Randomizer
from poppy_helpers.startups import startup_swordfight

SCALING = 5  # how many hundred


controller_att, controller_def = startup_swordfight("flogo2","flogo4")

rand = Randomizer()
norm = Normalizer()
inf = Inference("/home/florian/dev/pytorch-a2c-ppo-acktr/"
                "trained_models/ppo/"
                "ErgoFightStatic-Headless-Fencing-v0-180209225957.pt")

def get_observation():
    norm_pos_att = norm.normalize_pos(controller_att.get_pos_comp())
    norm_vel_att = norm.normalize_vel(controller_att.get_current_speed())
    norm_pos_def = norm.normalize_pos(controller_def.get_pos_comp())
    norm_vel_def = norm.normalize_vel(controller_def.get_current_speed())

    return np.hstack((norm_pos_att, norm_vel_att, norm_pos_def, norm_vel_def))


def window():
    def _create_circle(self, x, y, r, **kwargs):
        return self.create_oval(x - r, y - r, x + r, y + r, **kwargs)

    tk.Canvas.create_circle = _create_circle

    def keyup(e):
        canvas.create_circle(SCALING * 100, SCALING * 100, SCALING * 100, fill="black", width=1)

    def keydown(e):
        canvas.create_circle(SCALING * 100, SCALING * 100, SCALING * 100, fill="red", width=1)
        q.put_nowait((True,e.char))

    root = tk.Tk()
    canvas = tk.Canvas(root, width=SCALING * 200, height=SCALING * 200, borderwidth=0, highlightthickness=0, bg="black")
    canvas.grid()
    canvas.bind("<KeyPress>", keydown)
    canvas.bind("<KeyRelease>", keyup)
    canvas.pack()
    canvas.focus_set()
    root.mainloop()
    q.put_nowait(None)

def control():

    #FIXME: this is just copy-pasted... adjust and make work
    while True:
        for i in range(20):
            if i % 10 == 0:
                controller_def.goto_pos(rand.random_def_stance())
            action = inf.get_action(get_observation())
            action = np.clip(action, -1, 1)
            action = norm.denormalize_pos(action)
            controller_att.act(action, scaling=0.2)

        controller_att.safe_rest()
        controller_def.safe_rest()

        time.sleep(2)
    ##FIXME: end of fixme


    while True:
        print ("control loop running")
        time.sleep(3)
        if q.empty():
            continue
        item = q.get_nowait()
        q.task_done()
        if item is None:
            break
        else:
            pressrel = "keydown"
            if not item[0]:
                pressrel = "keyup"
            print ("got {}: {}".format(pressrel, item[1]))

q = queue.Queue()
threads = []

t = threading.Thread(target=window)
t.start()
threads.append(t)

t = threading.Thread(target=control)
t.start()
threads.append(t)

# block until all tasks are done
q.join()

for t in threads:
    t.join()