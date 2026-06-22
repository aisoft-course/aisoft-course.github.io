import threading
import time

counter = 0
N = 1000
lock = threading.Lock()

def task():
    global counter
    for _ in range(N):
        with lock:
            tmp = counter
            time.sleep(0.00001)
            counter = tmp + 1

threads = [threading.Thread(target=task) for _ in range(10)]
for t in threads:
    t.start()
for t in threads:
    t.join()

print(counter)
