# from digital_twin import DigitalTwin
# import time
# import pybullet as p 
import csv
import numpy as np
import matplotlib.pyplot as plt

iters = []
angle_error = []
angle_gt = []
angle_command = []
angle_main = []
angle_jpnp = []
with open('dt_debug.csv', 'r') as f:
    reader = csv.reader(f, delimiter=',')
    header = next(reader)
    for row in reader:
        iters.append(row[0])
        angle_error.append(row[1])
        angle_gt.append(row[2:8])
        angle_command.append(row[8:14])
        angle_main.append(row[14:20])
        angle_jpnp.append(row[20:26])        
iters = np.array(iters, dtype=np.float32)
angle_error = np.array(angle_error, dtype=np.float32)*180/np.pi
angle_gt = np.array(angle_gt, dtype=np.float32)*180/np.pi
angle_command = np.array(angle_command, dtype=np.float32)*180/np.pi
angle_main = np.array(angle_main, dtype=np.float32)*180/np.pi
angle_jpnp = np.array(angle_jpnp, dtype=np.float32)*180/np.pi


plt.figure(0)
plt.plot(iters, label='jpnp iterations')
plt.xlabel('steps')
plt.legend()

plt.figure(1)
plt.plot(angle_error, label='angle cos error')
plt.xlabel('steps')
plt.ylabel('angle [deg]')
plt.legend()
# plt.boxplot(angle_error)

plt.figure(2)
plt.subplot(2, 3, 1)
plt.plot(angle_gt[:,0], label='GT')
plt.plot(angle_command[:,0], label='command')
plt.plot(angle_main[:,0], label='main')
plt.plot(angle_jpnp[:,0], label='jpnp')
plt.title('Joint 1')
plt.xlabel('steps')
plt.ylabel('angle [deg]')
plt.legend()

plt.subplot(2, 3, 2)
plt.plot(angle_gt[:,1], label='GT')
plt.plot(angle_command[:,1], label='command')
plt.plot(angle_main[:,1], label='main')
plt.plot(angle_jpnp[:,1], label='jpnp')
plt.title('Joint 2')
plt.xlabel('steps')
plt.ylabel('angle [deg]')
plt.legend()

plt.subplot(2, 3, 3)
plt.plot(angle_gt[:,2], label='GT')
plt.plot(angle_command[:,2], label='command')
plt.plot(angle_main[:,2], label='main')
plt.plot(angle_jpnp[:,2], label='jpnp')
plt.title('Joint 3')
plt.xlabel('steps')
plt.ylabel('angle [deg]')
plt.legend()

plt.subplot(2, 3, 4)
plt.plot(angle_gt[:,3], label='GT')
plt.plot(angle_command[:,3], label='command')
plt.plot(angle_main[:,3], label='main')
plt.plot(angle_jpnp[:,3], label='jpnp')
plt.title('Joint 4')
plt.xlabel('steps')
plt.ylabel('angle [deg]')
plt.legend()

plt.subplot(2, 3, 5)
plt.plot(angle_gt[:,4], label='GT')
plt.plot(angle_command[:,4], label='command')
plt.plot(angle_main[:,4], label='main')
plt.plot(angle_jpnp[:,4], label='jpnp')
plt.title('Joint 5')
plt.xlabel('steps')
plt.ylabel('angle [deg]')
plt.legend()

plt.subplot(2, 3, 6)
plt.plot(angle_gt[:,5], label='GT')
plt.plot(angle_command[:,5], label='command')
plt.plot(angle_main[:,5], label='main')
plt.plot(angle_jpnp[:,5], label='jpnp')
plt.title('Joint 6')
plt.xlabel('steps')
plt.ylabel('angle [deg]')
plt.legend()
plt.show()