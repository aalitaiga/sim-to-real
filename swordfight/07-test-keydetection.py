import tkinter as tk

scaling = 6 # how many hundred

def _create_circle(self, x, y, r, **kwargs):
    return self.create_oval(x-r, y-r, x+r, y+r, **kwargs)
tk.Canvas.create_circle = _create_circle

def keyup(e):
    canvas.create_circle(scaling* 100, scaling*100, scaling*100, fill="black", width=1)
def keydown(e):
    canvas.create_circle(scaling*100, scaling*100, scaling*100, fill="red", width=1)

root = tk.Tk()
canvas = tk.Canvas(root, width=scaling*200, height=scaling*200, borderwidth=0, highlightthickness=0, bg="black")
canvas.grid()
canvas.bind("<KeyPress>", keydown)
canvas.bind("<KeyRelease>", keyup)
canvas.pack()
canvas.focus_set()
root.mainloop()