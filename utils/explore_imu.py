import polars as pl
import matplotlib.pyplot as plt

df = pl.read_parquet('../data/raw/imu/P01_LF.parquet')

print(df.columns)

plt.plot(df['gyro_y'])
plt.xlabel('Sample')
plt.ylabel('Gyro Y')
plt.title('Gyro Y Data')
plt.show()
