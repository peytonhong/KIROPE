# from digital_twin import DigitalTwin
# import time
# import pybullet as p 
import csv

with open('dt_debug.csv', 'r') as f:
    reader = csv.reader(f, delimiter=',')
    header = next(reader)
    for row in reader:
        iters, criteria, angle_gt, angle_command, angle_control = row
        print(angle_command)
        print(angle_control)
        break