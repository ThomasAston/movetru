# Simple script for processing large IMU CSV files by removing empty rows and unnecessary 
# columns before saving as parquet.
import polars as pl
from pathlib import Path

# Configuration
input_dir = Path("../data/raw/imu")
output_dir = Path("../data/raw/imu")
output_dir.mkdir(exist_ok=True)

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