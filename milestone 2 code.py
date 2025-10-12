import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import LabelEncoder
import joblib
import warnings
warnings.filterwarnings("ignore")
df = pd.read_csv("/content/air_pollution_data.csv", encoding='latin1')
print("Data loaded:", df.shape)
if 'date' not in df.columns:
    for c in df.columns:
        if 'date' in c.lower():
            df.rename(columns={c: 'date'}, inplace=True)
df['date'] = pd.to_datetime(df['date'], errors='coerce')
df.dropna(subset=['date'], inplace=True)
if 'no2' not in df.columns: df['no2'] = np.random.uniform(10, 150, len(df))
if 'so2' not in df.columns: df['so2'] = np.random.uniform(5, 80, len(df))
if 'pm2_5' not in df.columns: df['pm2_5'] = np.random.uniform(20, 200, len(df))
if 'humidity' not in df.columns: df['humidity'] = np.random.uniform(30, 90, len(df))
if 'dist_road_km' not in df.columns: df['dist_road_km'] = np.random.uniform(0.1, 5, len(df))
if 'dist_industry_km' not in df.columns: df['dist_industry_km'] = np.random.uniform(0.2, 8, len(df))
if 'dist_farmland_km' not in df.columns: df['dist_farmland_km'] = np.random.uniform(0.3, 10, len(df))
if 'season' not in df.columns: df['season'] = df['date'].dt.month % 12 // 3 + 1
def label_source(r):
    no2 = r['no2']
    so2 = r['so2']
    pm = r['pm2_5']
    hum = r['humidity']
    dr, di, dfarm = r['dist_road_km'], r['dist_industry_km'], r['dist_farmland_km']
    season = r['season']

    if dr < 0.5 and no2 > 80:
        return "Vehicular"
    elif di < 1 and so2 > 50:
        return "Industrial"
    elif dfarm < 2 and (hum < 40 or season == 3) and pm > 100:
        return "Agricultural"
    elif pm > 150 and hum < 40:
        return "Burning"
    else:
        return "Natural"

df['pollution_source'] = df.apply(label_source, axis=1)
print("Labels created:\n", df['pollution_source'].value_counts())

features = ['no2', 'so2', 'pm2_5', 'humidity',
            'dist_road_km', 'dist_industry_km', 'dist_farmland_km', 'season']
X = df[features]
y = df['pollution_source']

le = LabelEncoder()
y_enc = le.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(
    X, y_enc, test_size=0.2, random_state=42, stratify=y_enc)

print("Train/Test shapes:", X_train.shape, X_test.shape)

results = {}


dt = DecisionTreeClassifier(max_depth=8, random_state=42)
dt.fit(X_train, y_train)
y_pred_dt = dt.predict(X_test)
results["DecisionTree"] = accuracy_score(y_test, y_pred_dt)
print("\nDecision Tree Accuracy:", results["DecisionTree"])
print(classification_report(y_test, y_pred_dt, target_names=le.classes_))

# Random Forest
rf = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)
results["RandomForest"] = accuracy_score(y_test, y_pred_rf)
print("\nRandom Forest Accuracy:", results["RandomForest"])
print(classification_report(y_test, y_pred_rf, target_names=le.classes_))

# XGBoost 
if has_xgb:
    xgb = XGBClassifier(n_estimators=150, max_depth=6, learning_rate=0.05,
                        random_state=42, eval_metric="mlogloss")
    xgb.fit(X_train, y_train)
    y_pred_xgb = xgb.predict(X_test)
    results["XGBoost"] = accuracy_score(y_test, y_pred_xgb)
    print("\nXGBoost Accuracy:", results["XGBoost"])
    print(classification_report(y_test, y_pred_xgb, target_names=le.classes_))

best_model_name = max(results, key=results.get)
print(f"\n✅ Best model: {best_model_name} (Accuracy = {results[best_model_name]:.3f})")

best_model = {"DecisionTree": dt, "RandomForest": rf}.get(best_model_name, dt)
if has_xgb and best_model_name == "XGBoost":
    best_model = xgb

joblib.dump(best_model, "best_pollution_model.joblib")
joblib.dump(le, "label_encoder.joblib")

print("Model saved as best_pollution_model.joblib")
print("Label encoder saved as label_encoder.joblib")


df.to_csv("labeled_air_pollution_data.csv", index=False)
print("Labeled dataset saved as labeled_air_pollution_data.csv")
