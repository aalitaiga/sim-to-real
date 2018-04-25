import time

import keyboard

# ON LINUX AND MAC YOU NEED TO RUN THIS SCRIPT WITH SUDO


keyboard.add_hotkey(' ', print, args=['space was pressed'])

time.sleep(5)