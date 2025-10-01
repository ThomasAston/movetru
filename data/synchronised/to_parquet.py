import polars as pl
import pathlib
import sys

# Directory containing CSV files
csv_dir = pathlib.Path(__file__).parent

# Get all CSV files
csv_files = list(csv_dir.glob("*.csv"))

if not csv_files:
    print("No CSV files found in the directory.")
    sys.exit(0)

print(f"Found {len(csv_files)} CSV file(s) to convert...")

for csv_file in csv_files:
    try:
        print(f"Processing {csv_file.name}...", end=" ")
        
        # Read CSV file with extended schema inference
        # This ensures float columns aren't misidentified as integers
        df = pl.read_csv(csv_file, infer_schema_length=10000)
        
        # Validate dataframe is not empty
        if df.is_empty():
            print(f"⚠️  Warning: {csv_file.name} is empty")
            continue
        
        # Write to parquet
        parquet_file = csv_file.with_suffix('.parquet')
        df.write_parquet(parquet_file)
        
        print(f"✓ Converted to {parquet_file.name} ({len(df)} rows)")
        
    except Exception as e:
        print(f"✗ Error processing {csv_file.name}: {str(e)}")
        continue

print("\nConversion complete!")