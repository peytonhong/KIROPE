from digital_twin import DigitalTwin
import time

DT = DigitalTwin(headless=True)
start_time = time.time()
DT.forward()
print("elapsed time: {}".format(time.time()-start_time))