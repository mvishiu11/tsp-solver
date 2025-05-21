# utils/events.py
"""
Shared thread-safe primitives:
    * update_queue – algorithm ➜ UI messages
    * cancel_event – immediate global stop
    * pause_event  – blocks algorithm loops while cleared
"""
from __future__ import annotations
import threading
import queue

update_queue: "queue.Queue[dict]" = queue.Queue()

cancel_event = threading.Event()      # set()  ➜ all workers abort
pause_event = threading.Event()       # clear() ➜ workers wait
pause_event.set()                     # start un-paused
