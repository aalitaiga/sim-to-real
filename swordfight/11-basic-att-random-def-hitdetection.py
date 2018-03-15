from gevent import monkey

monkey.patch_all()
import gevent
import queue
import threading
import tkinter as tk
import time
import numpy as np
from pytorch_a2c_ppo_acktr.inference import Inference

from poppy_helpers.normalizer import Normalizer
from poppy_helpers.randomizer import Randomizer
from poppy_helpers.startups import startup_swordfight

from poppy_helpers.sound import RemoteSound

SCALING = 5  # how many hundred

controller_att, controller_def = startup_swordfight("flogo2", "flogo4")

rand = Randomizer()
norm = Normalizer()
inf = Inference("/home/florian/dev/pytorch-a2c-ppo-acktr/"
                "trained_models/ppo/"
                "ErgoFightStatic-Headless-Fencing-v0-180209225957.pt")
snd = RemoteSound("flogo4.local")


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

    root = tk.Tk()

    def keyup(e):
        canvas.create_circle(SCALING * 100, SCALING * 100, SCALING * 100, fill="black", width=1)

    def keydown(e):
        canvas.create_circle(SCALING * 100, SCALING * 100, SCALING * 100, fill="red", width=1)
        q.put((True, e.char))

    def gevent_loop_step():
        gevent.sleep()
        root.after(0, gevent_loop_step)

    canvas = tk.Canvas(root, width=SCALING * 200, height=SCALING * 200, borderwidth=0, highlightthickness=0, bg="black")
    canvas.grid()
    canvas.bind("<KeyPress>", keydown)
    canvas.bind("<KeyRelease>", keyup)
    canvas.pack()
    canvas.focus_set()
    root.after(50, gevent_loop_step)
    root.mainloop()
    q.put_nowait(None)


def flush_queue():
    while not q.empty():
        try:
            q.get(False)
        except queue.Empty:
            continue
        q.task_done()


def control():
    reset = True
    while True:
        if reset:
            controller_att.safe_rest()
            controller_def.safe_rest()
            time.sleep(2)
            controller_def.goto_pos(rand.random_def_stance())
            reset = False
            flush_queue()

        action = inf.get_action(get_observation())
        action = np.clip(action, -1, 1)
        action = norm.denormalize_pos(action)
        controller_att.act(action, scaling=0.2)

        if q.empty():
            continue
        item = q.get_nowait()
        q.task_done()
        if item is None:
            break
        else:
            if item[0] and item[1] == "d":  # this means keydown on key "d"
                snd.play("beep1")
                reset = True


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
