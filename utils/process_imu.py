# Simple script for processing large IMU CSV files by removing empty rows and unnecessary 
# columns before saving as parquet, then synchronizing left and right foot sensors.
import polars as pl
import numpy as np
from pathlib import Path

# Configuration
input_dir = Path("../data/raw/imu")
output_dir = Path("../data/raw/imu")
sync_output_dir = Path("../data/raw/imu")
output_dir.mkdir(exist_ok=True)
sync_output_dir.mkdir(exist_ok=True)

# Process all CSV files in the input directory
for csv_file in input_dir.glob("*.csv"):
    print(f"Processing {csv_file.name}...")
    
    # Read CSV, skipping the header lines but preserving column names
    # Line 6 (index 5) contains the actual column names
    df = pl.read_csv(csv_file, skip_rows=7, has_header=False)
    
    # Read the column names separately from line 6 (index 5)
    with open(csv_file, 'r') as f:
        lines = f.readlines()
        column_names = lines[5].strip().split(',')
    
    # Clean up column names - replace empty strings and handle duplicates
    cleaned_names = []
    for i, name in enumerate(column_names):
        if name.strip() == '':
            cleaned_names.append(f'col_{i}')
        else:
            cleaned_names.append(name.strip())
    
    # Assign proper column names
    df.columns = cleaned_names
    
    print(f"  Original shape: {df.shape}")
    
    # Remove rows where all sensor data is missing (keeping time-only rows with data)
    # In Polars, we'll filter out rows where all columns except the first are null
    df_clean = df.filter(~pl.all_horizontal(pl.col(df.columns[1:]).is_null()))
    
    # Keep only time, gyro, and accelerometer columns
    columns_to_keep = ['Time', 'Gyro X', 'Gyro Y', 'Gyro Z', 'Accel X', 'Accel Y', 'Accel Z']
    
    # Filter to only include columns that exist in the dataframe
    available_columns = [col for col in columns_to_keep if col in df_clean.columns]
    df_clean = df_clean.select(available_columns)
    
    print(f"  After cleaning: {df_clean.shape}")
    print(f"  Reduction: {(1 - len(df_clean)/len(df))*100:.1f}%")
    
    # Save as parquet for efficient storage
    output_file = output_dir / f"{csv_file.stem}.parquet"
    df_clean.write_parquet(output_file)
    
    # Show file size reduction
    original_size = csv_file.stat().st_size / (1024*1024)  # MB
    new_size = output_file.stat().st_size / (1024*1024)  # MB
    print(f"  Size: {original_size:.1f}MB -> {new_size:.1f}MB ({new_size/original_size*100:.1f}%)")
    print()

print("All files processed!")
print("\n" + "="*60)
print("Starting synchronization process...")
print("="*60 + "\n")

# Get all parquet files grouped by participant
parquet_files = sorted(output_dir.glob("*.parquet"))
participants = set([f.stem.split('_')[0] for f in parquet_files if '_' in f.stem])

for participant in sorted(participants):
    lf_file = output_dir / f"{participant}_LF.parquet"
    rf_file = output_dir / f"{participant}_RF.parquet"
    
    # Check if both left and right foot files exist
    if not (lf_file.exists() and rf_file.exists()):
        print(f"Skipping {participant} - missing LF or RF file")
        continue
    
    print(f"Synchronizing {participant}...")
    
    # Load the data
    df_lf = pl.read_parquet(lf_file)
    df_rf = pl.read_parquet(rf_file)
    
    # Convert Accel Z to numpy for spike detection
    lf_accel_z = df_lf['Accel Z'].to_numpy()
    rf_accel_z = df_rf['Accel Z'].to_numpy()
    
    # Find first big spike (d = mean + 3*std)
    lf_threshold = lf_accel_z.mean() + 3 * lf_accel_z.std()
    rf_threshold = rf_accel_z.mean() + 3 * rf_accel_z.std()
    # lf_threshold = 4
    # rf_threshold = 4
    # print(f"  LF threshold: {lf_threshold:.2f}"
    #       f", RF threshold: {rf_threshold:.2f}")
    lf_spike_idx = np.argmax(lf_accel_z > lf_threshold)
    rf_spike_idx = np.argmax(rf_accel_z > rf_threshold)
    
    print(f"  LF spike at index: {lf_spike_idx}")
    print(f"  RF spike at index: {rf_spike_idx}")
    
    # Synchronize all signals by aligning the first spike
    sync_length = min(len(df_lf) - lf_spike_idx, len(df_rf) - rf_spike_idx)
    
    # Sync left foot data
    lf_synced = df_lf[lf_spike_idx:lf_spike_idx + sync_length]
    
    # Sync right foot data
    rf_synced = df_rf[rf_spike_idx:rf_spike_idx + sync_length]
    
    # Reset time columns to start at zero
    if 'Time' in lf_synced.columns:
        lf_start_time = lf_synced['Time'][0]
        lf_synced = lf_synced.with_columns(
            (pl.col('Time') - lf_start_time).alias('Time')
        )
    
    if 'Time' in rf_synced.columns:
        rf_start_time = rf_synced['Time'][0]
        rf_synced = rf_synced.with_columns(
            (pl.col('Time') - rf_start_time).alias('Time')
        )
    
    print(f"  Synced length: {sync_length} samples")
    print(f"  Time columns reset to start at 0")
    
    # Save synchronized data
    lf_sync_file = sync_output_dir / f"{participant}_LF.parquet"
    rf_sync_file = sync_output_dir / f"{participant}_RF.parquet"
    
    lf_synced.write_parquet(lf_sync_file)
    rf_synced.write_parquet(rf_sync_file)
    
    print(f"  Saved to {lf_sync_file.name} and {rf_sync_file.name}")
    print()

print("Synchronization complete!")