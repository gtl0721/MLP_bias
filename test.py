import matplotlib.pyplot as plt
import numpy as np
path = "C:\\Users\\a8000\\OneDrive\\桌面\\HW#1\\test_y.dat"
path1 = "C:\\Users\\a8000\\OneDrive\\桌面\\HW#1\\OUT.dat"
test_data = []
out_data = []
error_data = []

with open(path,'r') as f :
    lines = f.readlines()
    for line in lines:
        value = [float(s) for s in line.split()]
        if(value != []):
            test_data.append(value)
with open(path1,'r') as f :
    lines = f.readlines()
    for line in lines:
        value = [float(s) for s in line.split()]
        if(value != []):
            out_data.append(value)

out_data = np.array(out_data)
test_data = np.array(test_data)
test_data = [test_data[i] + 0.4 for i in range(len(test_data))]
error_data = [out_data[i] - test_data[i] for i in range(len(out_data))]
max_err = 0
for i in range(len(out_data)):
    if(error_data[i] > max_err):
        max_err = error_data[i]

print("max_err = " , max_err)

plt.xlabel('time step')    
plt.plot(test_data, 'o', label='test_data', linewidth=1.0, color='red')
plt.plot(out_data, 'o', label='out_data', linewidth=1.0, color='blue')
plt.plot(error_data, 'o', label='error_data', linewidth=1.0, color='green')
plt.legend(['test_data', 'out_data', 'error_data'])
plt.show()