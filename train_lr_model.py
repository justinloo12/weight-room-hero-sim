"""
Train logistic regression model on historical homerun data.
This replaces the hand-coded z-score approach with actual predictive power.
"""
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import pickle
 
print("Loading enriched homerun data...")
df = pd.read_csv('homerun_data_enriched.csv')
hitter_profiles = pd.read_csv('hitter_profiles.csv')
pitcher_profiles = pd.read_csv('pitcher_profiles.csv')
 
# Target variable: whether this at-bat resulted in a homerun
df['target'] = (df['events'] == 'home_run').astype(int)
 
print(f"Total at-bats: {len(df):,}")
print(f"Total home runs: {df['target'].sum():,}")
print(f"Base HR rate: {df['target'].mean():.2%}")
 
# Merge platoon splits from hitter profiles (select specific columns to avoid conflicts)
df = df.merge(
    hitter_profiles[['batter', 'h_hr_vs_rhp', 'h_hr_vs_lhp']],
    on='batter',
    how='left',
    suffixes=('', '_profile')
)
 
print(f"Merged with hitter profiles, now {len(df.columns)} columns")
 
# Encode wind direction as circular features and ballpark factors
def encode_wind_dir(direction):
    """Convert wind direction to numeric cardinal direction (0-360 mapped to 0-1)"""
    if pd.isna(direction):
        return 0.5
    wind_map = {'N': 0, 'NNE': 22.5, 'NE': 45, 'ENE': 67.5, 'E': 90, 'ESE': 112.5, 'SE': 135, 
                'SSE': 157.5, 'S': 180, 'SSW': 202.5, 'SW': 225, 'WSW': 247.5, 'W': 270, 
                'WNW': 292.5, 'NW': 315, 'NNW': 337.5}
    return wind_map.get(str(direction).upper(), 180) / 360.0
 
# Ballpark HR rate factors (relative to league average of 2.41%)
ballpark_factors = {
    'NYY': 1.25,  # Yankee Stadium - short porch
    'COL': 1.35,  # Coors Field - high altitude
    'TEX': 1.15,  # Globe Life - hitter friendly
    'HOU': 1.10,  # MMP - slightly elevated
    'ARI': 1.08,  # Chase Field - moderate
    'OAK': 0.92,  # Oakland - pitcher friendly
    'PIT': 0.88,  # PNC - pitcher friendly
    'SD': 0.85,   # Petco - pitcher friendly
}
 
# Create derived features
df['wind_dir_encoded'] = df['wind_dir'].apply(encode_wind_dir)
df['ballpark_hr_factor'] = df['ballpark_code'].map(ballpark_factors).fillna(1.0)
 
# Platoon-matched HR rate: use the appropriate rate based on pitcher handedness
df['platoon_matched_hr_rate'] = np.where(
    df['pitcher_right'] == 1,
    df['h_hr_vs_rhp'],  # Batter vs RHP
    df['h_hr_vs_lhp']   # Batter vs LHP
)
 
print("Platoon and ballpark features added.")
 
# Select features for the model - ONLY use features available at prediction time
feature_cols = [
    # Hitter profile stats (available pre-game)
    'h_barrel_pct',
    'h_exit_velo',
    'h_hard_hit_pct',
    'h_pull_air_pct',
    'h_hr_rate',
    'h_launch_angle',
    
    # Platoon splits (available pre-game)
    'platoon_matched_hr_rate',
    
    # Pitcher profile stats (available pre-game)
    'p_hr_rate_allowed',
    'p_barrel_pct_allowed',
    'p_hard_hit_pct_allowed',
    'p_exit_velo_allowed',
    'p_pull_air_pct_allowed',
    
    # Context: ballpark and weather (available pre-game)
    'is_coors',
    'ballpark_hr_factor',
    'temp_f',
    'humidity',
    'wind_speed',
    'wind_dir_encoded',
    'batter_right',
    'pitcher_right',
]
 
# Handle missing values and prepare features
X = df[feature_cols].copy()
y = df['target'].copy()
 
# Drop rows with missing target
mask = y.notna()
X = X[mask]
y = y[mask]
 
print(f"\nRows with valid target: {len(X):,}")
 
# Fill missing values with column median (reasonable for stats)
for col in X.columns:
    missing = X[col].isna().sum()
    if missing > 0:
        X[col].fillna(X[col].median(), inplace=True)
        print(f"  {col}: filled {missing:,} missing with median")
 
# Drop any rows still with NaN
X = X.dropna()
y = y[X.index]
 
print(f"Rows after removing NaN: {len(X):,}")
print(f"HR rate in training set: {y.mean():.2%}")
 
# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
 
# Train logistic regression
print("\nTraining logistic regression...")
lr = LogisticRegression(
    max_iter=1000,
    random_state=42,
    class_weight='balanced',  # account for class imbalance
    solver='lbfgs'
)
lr.fit(X_scaled, y)
 
print(f"Model accuracy: {lr.score(X_scaled, y):.3f}")
 
# Show feature importance (coefficients)
print("\nTop 10 predictive features:")
feature_importance = pd.DataFrame({
    'feature': feature_cols,
    'coefficient': lr.coef_[0],
    'abs_coef': np.abs(lr.coef_[0])
}).sort_values('abs_coef', ascending=False)
 
for idx, row in feature_importance.head(10).iterrows():
    direction = "↑ helps HR" if row['coefficient'] > 0 else "↓ reduces HR"
    print(f"  {row['feature']}: {row['coefficient']:+.4f} {direction}")
 
# Save model and scaler
print("\nSaving model and scaler...")
with open('lr_model.pkl', 'wb') as f:
    pickle.dump(lr, f)
with open('lr_scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)
with open('lr_features.pkl', 'wb') as f:
    pickle.dump(feature_cols, f)
 
print("Done! Models saved.")