import time
import os
import serial
import numpy as np

# Adjust the following variables to your specific case
#mcu_port = "/dev/ttyACM0" # Nano 33
mcu_port = "/dev/ttyUSB0" # Serial port of the microcontroller
data_path = "./data/cifar-10-batches-bin/test_batch.bin" # Location of the data file
baud_rate = 460800  # baud rate of the serial comms
image_size = 3073



'''
# uncomment if you want to check your port name
import serial.tools.list_ports

ports = serial.tools.list_ports.comports()
for p in ports:
    print(p)
'''

def load_data_file(file_path):
    if os.path.exists(file_path):
        data = np.load(file_path, allow_pickle=True).item()
        j = len(data['inference'])
        resume = input(f"The data file '{file_path}' already exists. Do you want to resume from the last run (start from image number {j+1})? (y/n): ").lower()

        if resume == 'y':
            return j, data

    return 0, {'inference': np.zeros(0, dtype=int),
                    'times': np.zeros(0),
                    'logReg': np.zeros(0, dtype=int),
                    'tabulated': np.zeros(0, dtype=int)}


# Check if there exist a data file to store the results so that we can recover previous progress
file_path = "data_file.npy"
# Load existing data file or get the number of images for a new run
j, previous_run = load_data_file(file_path)
starting_at = j

inp = input("Enter the total number of images (press enter to use the entire data set): ")
if inp == None:
    num_images = len(data) / image_size

else:
    num_images = int(inp)




# Initialize your arrays with the appropriate size
inference = np.concatenate([previous_run['inference'], np.zeros(num_images, dtype=int)])
times = np.concatenate([previous_run['times'], np.zeros(num_images)])
logReg = np.concatenate([previous_run['logReg'], np.zeros(num_images, dtype=int)])
tabulated = np.concatenate([previous_run['tabulated'], np.zeros(num_images, dtype=int)])


# Open the serial port
ser = serial.Serial(mcu_port, baud_rate)#, dsrdtr=True) 
time.sleep(1)

# Read the binary file
with open(data_path, 'rb') as file:
    data = file.read()


# write to serial communication the images, starting from the last found in the file "data_file.npy" up to the selected
# number of runs selected by the user.
for i in range(j*image_size, len(data), image_size):
    ser.write(data[i+1:i+image_size])
    time.sleep(0.1)

    # the program will wait for the ESP32 to send a signal with the character "#" that announce the program is going
    # to send the results of the inference. Then, the program will read the content of the serial communication 
    start_marker = ''
    while not start_marker:
        start_marker = ser.read().decode('utf-8')
        if start_marker == '#':
            break

    # If there is content in the buffer, read it and store it in the corresponding arrays
    if ser.in_waiting > 0:
        received_data = ser.readline().decode('utf-8').strip()

    data_elements = received_data.split(',')

    # format and store the results
    scores = [int(e) for e in data_elements[:10]]
    inference[j] = scores.index(max(scores))
    tabulated[j] = data[i]
    logReg[j] = int(data_elements[10])
    times[j] = float(data_elements[11])/1000000
    print(f'Processing image {j-starting_at} out of {num_images}', end='\r')


    j += 1
    if j >= num_images + starting_at:
        break

# we store the results so that we can resume the program later from this point.
np.save(file_path, {'inference': inference, 'times': times, 'logReg': logReg, 'tabulated': tabulated})

# we now calculate the precision of the inference
goods = 0   # number of correct predictions of the complete system (TinyML + Logistic Regression)
noOff = 0   # number of times the Logistic Regression correctly predicts the result from TinyML was wrong
bads = 0    # number of times the complete system (TinyML + Logistic Regression) was wrong
for i in range(j):
    if (logReg[i] == 0 or (logReg[i] == 1 and inference[i] == tabulated[i])):
    # Either an offload (assume the big model has 100% accuracy) or LR correctly accepts the result of TinyML
        goods += 1
    else:
    # TinyML was wrong and LR was wrong
        bads += 1

    if logReg[i] == 0 and inference[i] != tabulated[i]:
    # TinyML was wrong and LR correctly predicts the result of TinyML was wrong
        noOff += 1


print(f'\n\nOut of {j} images\n__________________\nGood predictions: {goods}\nBad predictions: {bads}\nPrecision: {goods/j*100}\nNumber of offloads: {np.sum(logReg == 0)}')
print(f'Of those offloads, {noOff} were correctly predicted by LR to be a wrong inference\nprec without offload: {(goods-noOff)/j*100}\n')
print(f'Avg inference time: {np.sum(times)/len(times)}')

ser.close()
