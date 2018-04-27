import threading
try:
    from queue import Queue
except ImportError:
    from Queue import Queue
import time

import keyboard


class Keylogger(threading.Thread):
  def __init__(self, q, ks):
    threading.Thread.__init__(self)
    self._queue = q
    self._killswitch = ks
    keyboard.add_hotkey(' ', self.add)
    # keyboard.add_hotkey(' ', self.add, args=['space was pressed'])

  def add(self):
      self._queue.put("space")

  def run(self):
    while True:
        # queue.get() blocks the current thread until
        # an item is retrieved.
        msg = self._killswitch.get()
        # Checks if the current message is
        # the "Poison Pill"
        if isinstance(msg, str) and msg == 'quit':
            # if so, exists the loop
            break
        # "Processes" (or in our case, prints) the queue item
        print ("I'm a thread, and I received %s!!" % msg)

        # Always be friendly!
    print ('Bye byes!')


# Queue is used to share items between
# the threads.
queue_ = Queue()
killswitch = Queue()

# Create an instance of the worker
worker = Keylogger(queue_, killswitch)
# start calls the internal run() method to
# kick off the thread
worker.start()

# variable to keep track of when we started
start_time = time.time()
# While under 5 seconds..
while time.time() - start_time < 5:

    spaces = []
    while not queue_.empty():
        spaces.append(queue_.get())

    print("received {} spaces: {}".format(len(spaces), spaces))

    time.sleep(1)

killswitch.put("quit")
worker.join()
