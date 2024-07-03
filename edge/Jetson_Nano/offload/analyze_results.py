'''
Take the file 'latencies.txt and' get the average and standard deviation of the latencies.
We check if there is any anomaly because some measurements, especially in wireless communications,
can show anomalies on the first iterations.
'''

import numpy as np
# Read the latencies from the file
latencies = np.loadtxt('latencies.txt')

# Calculate the average and standard deviation of the latencies
average_latency = np.mean(latencies)
std_deviation = np.std(latencies)

# Print the results
print("Average latency:", average_latency)
print(f'Standard deviation: {std_deviation}  (or {std_deviation/average_latency*100:.2f}% of the average)')

# Check if there is any anomaly. Count how many latencies are above the average + 3 * std_deviation
# and print the biggest anomaly
num_anomalies = 0
biggest_anomaly = 0.0
for latency in latencies:
    if latency > average_latency + 3 * std_deviation:
        num_anomalies += 1
        if latency > biggest_anomaly:
            biggest_anomaly = latency

print(f'{num_anomalies} samples were 3 standard deviations above the average')
print(f'The biggest anomaly is {biggest_anomaly} ms, which is {biggest_anomaly - average_latency} ms above the average ({(biggest_anomaly - average_latency) / average_latency * 100:.2f}%)')
