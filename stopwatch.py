import time

class Stopwatch:
    def __init__(self):
        self.start_time = 0

    def start(self):
        self.start_time = time.time()

    def get_time(self):
        print(time.time() - self.start_time)