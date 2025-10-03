import polars as pl
import matplotlib.pyplot as plt
import numpy as np

df_lf = pl.read_parquet('../data/raw/imu/P05_LF.parquet')
df_rf = pl.read_parquet('../data/raw/imu/P05_RF.parquet')

# Convert to numpy arrays for easier processing
lf_accel_z = df_lf['Accel Z'].to_numpy()
rf_accel_z = df_rf['Accel Z'].to_numpy()

# Find first big spike (e.g., threshold = mean + 3*std)
lf_threshold = lf_accel_z.mean() + 3 * lf_accel_z.std()
rf_threshold = rf_accel_z.mean() + 3 * rf_accel_z.std()

lf_spike_idx = np.argmax(lf_accel_z > lf_threshold)
rf_spike_idx = np.argmax(rf_accel_z > rf_threshold)

# Synchronize all signals by aligning the first spike
sync_length = min(len(df_lf) - lf_spike_idx, len(df_rf) - rf_spike_idx)

# Sync left foot data
lf_synced = df_lf[lf_spike_idx:lf_spike_idx + sync_length]

# Sync right foot data
rf_synced = df_rf[rf_spike_idx:rf_spike_idx + sync_length]

# Extract gyroscope Y values from synced data
lf_gyro_y = lf_synced['Gyro Y'].to_numpy()
rf_gyro_y = rf_synced['Gyro Y'].to_numpy()

fig, axs = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

axs[0].plot(lf_gyro_y)
axs[0].set_title('Left Foot - Gyro Y (Synced)')
axs[0].set_ylabel('Gyro Y')
axs[0].set_xlim([440000, 460000])  # Zoom in on a specific range for clarity

axs[1].plot(rf_gyro_y)
axs[1].set_title('Right Foot - Gyro Y (Synced)')
axs[1].set_xlabel('Sample')
axs[1].set_ylabel('Gyro Y')

plt.tight_layout()
plt.show()