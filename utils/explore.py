import pandas as pd
import matplotlib.pyplot as plt
import ezc3d
import numpy as np

# Read CSV data
df = pd.read_csv('../data/synchronised/P01_S01_FastGait_02.csv')

# # Plot gyro data from CSV
fig = plt.figure(figsize=(12, 8))
# plt.plot(df['P6_RF_gyro_x'], label='x rotational velocity')
plt.plot(df['P6_LF_gyro_y'], label='pitch velocity')
# plt.plot(df['P6_RF_gyro_z'], label='z rotational velocity')
plt.legend()
plt.title('Gyroscope Data from CSV')
plt.show()

# df = pd.read_csv('../data/processed_P01_S01_LF.csv')
# fig = plt.figure(figsize=(12, 8))
# # plt.plot(df['P6_RF_gyro_x'], label='x rotational velocity')
# plt.plot(df['gyro_y_raw'][:], label='pitch velocity')
# # plt.plot(df['P6_RF_gyro_z'], label='z rotational velocity')
# plt.legend()
# plt.title('Gyroscope Data from CSV')
# plt.show()