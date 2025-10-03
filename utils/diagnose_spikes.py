import polars as pl
import matplotlib.pyplot as plt
import numpy as np

# Check P06 and P08 spike detection
participants = ['P06', 'P08']

for participant in participants:
    print(f"\n{'='*60}")
    print(f"Analyzing {participant}")
    print('='*60)
    
    df_lf = pl.read_parquet(f'../data/raw/imu/{participant}_LF.parquet')
    df_rf = pl.read_parquet(f'../data/raw/imu/{participant}_RF.parquet')
    
    lf_accel_z = df_lf['Accel Z'].to_numpy()
    rf_accel_z = df_rf['Accel Z'].to_numpy()
    
    # Current threshold method
    lf_threshold = lf_accel_z.mean() + 3 * lf_accel_z.std()
    rf_threshold = rf_accel_z.mean() + 3 * rf_accel_z.std()
    
    lf_spike_idx = np.argmax(lf_accel_z > lf_threshold)
    rf_spike_idx = np.argmax(rf_accel_z > rf_threshold)
    
    print(f"LF: mean={lf_accel_z.mean():.2f}, std={lf_accel_z.std():.2f}, threshold={lf_threshold:.2f}")
    print(f"RF: mean={rf_accel_z.mean():.2f}, std={rf_accel_z.std():.2f}, threshold={rf_threshold:.2f}")
    print(f"LF spike at index: {lf_spike_idx} (value: {lf_accel_z[lf_spike_idx]:.2f})")
    print(f"RF spike at index: {rf_spike_idx} (value: {rf_accel_z[rf_spike_idx]:.2f})")
    
    # Visualize
    fig, axs = plt.subplots(2, 1, figsize=(15, 8), sharex=True)
    
    # Plot full signal
    axs[0].plot(lf_accel_z, label='LF Accel Z', alpha=0.7)
    axs[0].axhline(lf_threshold, color='r', linestyle='--', label=f'Threshold ({lf_threshold:.2f})')
    axs[0].axvline(lf_spike_idx, color='g', linestyle='--', label=f'Detected spike ({lf_spike_idx})')
    axs[0].set_title(f'{participant} - Left Foot Accel Z')
    axs[0].set_ylabel('Accel Z')
    axs[0].legend()
    axs[0].grid(True, alpha=0.3)
    
    axs[1].plot(rf_accel_z, label='RF Accel Z', alpha=0.7)
    axs[1].axhline(rf_threshold, color='r', linestyle='--', label=f'Threshold ({rf_threshold:.2f})')
    axs[1].axvline(rf_spike_idx, color='g', linestyle='--', label=f'Detected spike ({rf_spike_idx})')
    axs[1].set_title(f'{participant} - Right Foot Accel Z')
    axs[1].set_xlabel('Sample')
    axs[1].set_ylabel('Accel Z')
    axs[1].legend()
    axs[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'../data/raw/imu/{participant}_spike_diagnosis.png', dpi=150)
    print(f"Saved visualization to {participant}_spike_diagnosis.png")
    
plt.show()
