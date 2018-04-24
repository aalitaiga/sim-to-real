import queue
import threading
import tkinter as tk
import time

SCALING = 6  # how many hundred


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