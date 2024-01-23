import time

class Stopwatch:
    def __init__(self):
        self.start_time = 0
        self.elapsed_time = 0

    def start(self):
        self.start_time = time.perf_counter()
        self.elapsed_time = 0

    def pause(self):
        self.elapsed_time += self.get_time()

    def resume(self):
        self.start_time = time.perf_counter()


    def print_time(self):
        print(time.perf_counter() - self.start_time)

    def get_time(self):
        return time.perf_counter() - self.start_time

    def get_elapsed_time(self):
        return self.elapsed_time + self.get_time()


if __name__=="__main__":
    stopwatch = Stopwatch()
    stopwatch.start()
    time.sleep(1.0)
    print("Measure | Expected")
    print(stopwatch.get_elapsed_time(), 1.0)
    stopwatch.pause()
    time.sleep(1.0)
    stopwatch.resume()
    time.sleep(1.0)
    print(stopwatch.get_elapsed_time(), 2.0)
    print(stopwatch.get_time(), 1.0)