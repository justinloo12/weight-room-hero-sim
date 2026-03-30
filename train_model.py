import pandas as pd
import numpy as np
import pickle
import time
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import CalibratedClassifierCV
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, brier_score_loss
 
start_time = time.time()

print("Loading enriched data...")
df = pd.read_csv("homerun_data_enriched.csv")
print(f"Rows: {len(df):,} | HRs: {df['is_homerun'].sum():,} | HR rate: {df['is_homerun'].mean():.3f}")
print(f"  Load time: {time.time() - start_time:.1f}s")
 
# ── Feature groups (equal weight within each group) ───────────
# StandardScaler will normalize ALL features to mean=0, std=1
# so no single feature can dominate just because of its scale.
# Outliers naturally contribute more because their z-scores are larger.
# No raw at-bat features — only what's known BEFORE the game.
 
HITTER_FEATURES = [
    # Core HR shape
    "h_barrel_pct",
    "h_exit_velo",
    "h_hard_hit_pct",
    "h_launch_angle",
    "h_sweet_spot_pct",
    "h_pull_air_pct",
    "h_hr_rate",
    "h_hr_contact_score",
    "h_lifted_power_score",
    # Handedness backbone
    "h_hr_rate_vs_hand",
    "h_xwoba_vs_hand",
    "h_barrel_pct_vs_hand",
    "h_hard_hit_pct_vs_hand",
    "h_launch_angle_vs_hand",
    "h_sweet_spot_pct_vs_hand",
    "h_pull_air_pct_vs_hand",
    "h_hr_contact_score_vs_hand",
    "h_lifted_power_score_vs_hand",
]
 
PITCHER_FEATURES = [
    # Overall vulnerability
    "p_hr_rate_allowed",
    "p_barrel_pct_allowed",
    "p_hard_hit_pct_allowed",
    "p_exit_velo_allowed",
    "p_launch_angle_allowed",
    "p_sweet_spot_pct_allowed",
    "p_pull_air_pct_allowed",
    "p_hr_contact_risk",
    "p_lift_damage_risk",
    # Side-specific weakness
    "p_hr_rate_allowed_vs_side",
    "p_barrel_pct_allowed_vs_side",
    "p_hard_hit_pct_allowed_vs_side",
    "p_exit_velo_allowed_vs_side",
    "p_launch_angle_allowed_vs_side",
    "p_sweet_spot_pct_allowed_vs_side",
    "p_pull_air_pct_allowed_vs_side",
    "p_hr_contact_risk_vs_side",
    "p_lift_damage_risk_vs_side",
]

MATCHUP_FEATURES = [
    "m_barrel_matchup_score",
    "m_lift_matchup_score",
    "m_hr_contact_matchup",
    "m_lifted_power_matchup",
    "m_handed_hr_matchup",
    "m_handed_contact_matchup",
    "m_handed_lift_matchup",
    "m_pitch_hr_matchup",
    "m_pitch_contact_matchup",
    "m_pitch_ev_matchup",
    "m_sweet_spot_contact_edge",
    "m_zone_attack_edge",
]

CONTEXT_FEATURES = [
    "batter_right", "pitcher_right", "ballpark_code",
    "is_coors", "temp_f", "humidity", "wind_speed", "wind_dir",
]
 
all_features = HITTER_FEATURES + PITCHER_FEATURES + MATCHUP_FEATURES + CONTEXT_FEATURES
features = [f for f in all_features if f in df.columns]
 
print(f"\nUsing {len(features)} features:")
for name, group in [("Hitter", HITTER_FEATURES), ("Pitcher", PITCHER_FEATURES), ("Matchup", MATCHUP_FEATURES), ("Context", CONTEXT_FEATURES)]:
    found   = [f for f in group if f in df.columns]
    missing = [f for f in group if f not in df.columns]
    print(f"  {name}: {len(found)}/{len(group)}", end="")
    if missing: print(f"  (missing: {missing})", end="")
    print()
 
X = df[features].fillna(0)
y = df["is_homerun"]
 
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"\nTrain: {len(X_train):,} | Test: {len(X_test):,}")
print(f"Train HR rate: {y_train.mean():.3f} | Test HR rate: {y_test.mean():.3f}")
prep_done = time.time()
print(f"  Prep time: {prep_done - start_time:.1f}s")
 
# ── Build pipeline ─────────────────────────────────────────────
# StandardScaler: normalizes every feature to mean=0, std=1.
#   - Each feature gets EQUAL footing regardless of raw scale.
#   - Outliers have LARGER z-scores so they push predictions harder.
#
# GradientBoosting: learns non-linear interactions between features.
#   - Shallower trees (max_depth=4) prevent any one feature from dominating.
#   - More trees (n_estimators=300) but with low learning rate = stable.
#
# CalibratedClassifierCV: maps raw scores to realistic probabilities.
#   - Prevents the "everyone is 90%" problem.
#   - Outputs calibrated probabilities that match real-world HR rates.
 
base_model = GradientBoostingClassifier(
    n_estimators=180,
    max_depth=4,           # shallow = balanced across features
    learning_rate=0.05,
    min_samples_leaf=30,
    subsample=0.8,
    random_state=42,
)
 
pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("model",  CalibratedClassifierCV(base_model, method="sigmoid", cv=3)),
])
 
print("\nTraining (3-6 min)...")
fit_start = time.time()
pipeline.fit(X_train, y_train)
fit_end = time.time()
print(f"  Fit time: {fit_end - fit_start:.1f}s")
 
# ── Evaluate ──────────────────────────────────────────────────
y_prob = pipeline.predict_proba(X_test)[:, 1]
auc    = roc_auc_score(y_test, y_prob)
brier  = brier_score_loss(y_test, y_prob)
 
print(f"\nAUC:         {auc:.4f}  (higher = better, 0.5 = random)")
print(f"Brier Score: {brier:.4f} (lower = better, measures probability accuracy)")
print(f"Avg predicted HR prob: {y_prob.mean()*100:.2f}%  (real-world avg: ~{y_test.mean()*100:.2f}%)")
 
# Show probability distribution
for threshold in [0.03, 0.05, 0.08, 0.10, 0.15, 0.20]:
    pct = (y_prob >= threshold).mean() * 100
    print(f"  Players with >{threshold*100:.0f}% prob: {pct:.1f}% of at-bats")
 
# Feature importance (from the base model inside calibrator)
try:
    scaler = pipeline.named_steps["scaler"]
    calibrated = pipeline.named_steps["model"]
    importances = np.mean([
        est.base_estimator.feature_importances_
        for est in calibrated.calibrated_classifiers_
    ], axis=0)
    imp_series = pd.Series(importances, index=features).sort_values(ascending=False)
    print("\nTop 15 Features by Importance:")
    for feat, imp in imp_series.head(15).items():
        print(f"  {feat:<35} {imp:.4f}")
except Exception:
    print("\n(Feature importances not available for calibrated model)")
 
# ── Save ─────────────────────────────────────────────────────
pickle.dump(pipeline, open("hr_model.pkl", "wb"))
pd.DataFrame({"feature": features}).to_csv("model_features.csv", index=False)
print("\nSaved: hr_model.pkl, model_features.csv")
print(f"Total runtime: {time.time() - start_time:.1f}s")
print("Run dashboard.py!")
