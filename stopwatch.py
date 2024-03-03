
from util import Stopwatch

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