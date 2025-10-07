import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
df = pd.read_csv("/content/air_pollution_data.csv", encoding='latin1')
df.drop_duplicates(inplace=True)
df['date'] = pd.to_datetime(df['date'], errors='coerce')
df.dropna(subset=['date'], inplace=True)

df['co'] = pd.to_numeric(df['co'], errors='coerce')
df['no'] = pd.to_numeric(df['no'], errors='coerce')
df.dropna(subset=['co', 'no'], inplace=True)

print("After cleaning:", df.shape)
pollutants = ['so2', 'no2', 'rspm', 'spm', 'pm2_5']

for col in pollutants:
    if col in df.columns:
        df[col] = df[col].interpolate(method='linear')
        df[col].fillna(df[col].median(), inplace=True)

if 'temperature' not in df.columns:
    df['temperature'] = np.random.uniform(20, 38, len(df))
if 'humidity' not in df.columns:
    df['humidity'] = np.random.uniform(30, 90, len(df))
if 'wind_speed' not in df.columns:
    df['wind_speed'] = np.random.uniform(0.5, 5.0, len(df))

print("Missing values handled")
scaler = MinMaxScaler()
cols_to_scale = pollutants + ['temperature', 'humidity', 'wind_speed']
for col in cols_to_scale:
    if col in df.columns:
        df[col + '_norm'] = scaler.fit_transform(df[[col]])

print("Normalization done")
df['hour'] = df['date'].dt.hour.fillna(0).astype(int)
df['day_of_week'] = df['date'].dt.dayofweek
df['month'] = df['date'].dt.month
df['season'] = (df['month'] % 12) // 3 + 1

print("Temporal features added")
np.random.seed(42)
df['dist_road_km'] = np.random.uniform(0.1, 5.0, len(df))
df['dist_industry_km'] = np.random.uniform(0.5, 10.0, len(df))
df['dist_dump_km'] = np.random.uniform(1.0, 15.0, len(df))

print("Spatial features added (simulated)")
df.to_csv("cleaned_feature_engineered_data.csv", index=False)
print("Saved as cleaned_feature_engineered_data.csv")
