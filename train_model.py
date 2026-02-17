"""
train_model.py
==============

Script to train and save the World Development Clustering Model.

This script should be run after you've prepared your data and executed
all the preprocessing steps in your notebook. It will train a KMeans
clustering model and save it for deployment.

HOW TO RUN:
-----------
python train_model.py

"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.impute import SimpleImputer
from sklearn.metrics import silhouette_score
from world_development_model import WorldDevelopmentClusteringModel

print("=" * 80)
print(" WORLD DEVELOPMENT CLUSTERING MODEL - TRAINING SCRIPT")
print("=" * 80)
print()

# ============================================================================
# STEP 1: LOAD & PREPROCESS DATA
# ============================================================================

print("Step 1: Loading and preprocessing data...")
print("-" * 80)

try:
    # Load data
    df = pd.read_excel("World_development_mesurement.xlsx")
    print(f"  Loaded {len(df)} records from World_development_mesurement.xlsx")
    print(f"  Columns: {list(df.columns)}")
    print()
    
except FileNotFoundError:
    print("   Error: World_development_mesurement.xlsx not found!")
    print("   Make sure the file is in the same directory as this script.")
    exit(1)

# ============================================================================
# STEP 2: CLEAN & PREPARE DATA
# ============================================================================

print("Step 2: Cleaning and preparing data...")
print("-" * 80)

df_clean = df.copy()

# Identify categorical columns
cat_cols = df_clean.select_dtypes(include='object').columns

# Convert currency and percentage columns to numeric
numeric_like_cols = []
for col in cat_cols:
    if col != 'Country':
        df_clean[col] = (
            df_clean[col]
            .astype(str)
            .str.replace('$', '', regex=False)
            .str.replace(',', '', regex=False)
            .str.replace('%', '', regex=False)
            .str.strip()
        )
        df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
        numeric_like_cols.append(col)

print(f" Converted {len(numeric_like_cols)} columns to numeric")
print()

# Get numeric columns
num_cols = df_clean.select_dtypes(include=['int64', 'float64']).columns

# Impute missing values using median
print("Step 3: Imputing missing values...")
print("-" * 80)

imputer = SimpleImputer(strategy='median')
df_clean[num_cols] = imputer.fit_transform(df_clean[num_cols])
print(f" Imputed missing values using median strategy")
print()

# ============================================================================
# STEP 4: LOG TRANSFORMATION
# ============================================================================

print("Step 4: Applying log transformations...")
print("-" * 80)

df_model = df_clean.drop(columns=['Country'])

log_cols = [
    'GDP',
    'CO2 Emissions',
    'Energy Usage',
    'Health Exp/Capita',
    'Tourism Inbound',
    'Tourism Outbound'
]

for col in log_cols:
    if col in df_model.columns:
        df_model[col] = np.log1p(df_model[col])
        print(f"   Log-transformed: {col}")

print()

# ============================================================================
# STEP 5: STANDARDIZE FEATURES
# ============================================================================

print("Step 5: Standardizing features...")
print("-" * 80)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(df_model)
X_scaled = pd.DataFrame(X_scaled, columns=df_model.columns)

print(f"  Standardized {X_scaled.shape[1]} features")
print(f"  Shape: {X_scaled.shape}")
print()

# ============================================================================
# STEP 6: TRAIN K-MEANS MODEL
# ============================================================================

print("Step 6: Training K-Means model...")
print("-" * 80)

best_kmeans = KMeans(n_clusters=3, random_state=42)
kmeans_labels = best_kmeans.fit_predict(X_scaled)

sil_kmeans = silhouette_score(X_scaled, kmeans_labels)

print(f"  K-Means model trained successfully!")
print(f"  Number of clusters: 3")
print(f"  Silhouette Score: {sil_kmeans:.4f}")
print()

# ============================================================================
# STEP 7: SAVE MODEL
# ============================================================================

print("Step 7: Saving model...")
print("-" * 80)

# Initialize and train the model class
model = WorldDevelopmentClusteringModel(n_clusters=3, random_state=42)
model.fit(df_clean)

# Save to disk
model.save('world_development_kmeans.pkl')
print()
print("=" * 80)
print("  MODEL TRAINING COMPLETE!")
print("=" * 80)
print()
print(" Model saved as: world_development_kmeans.pkl")
print(f"Silhouette Score: {sil_kmeans:.4f}")
print()
print("NEXT STEPS:")
print("-" * 80)
print("1. Run the Streamlit app:")
print("   streamlit run streamlit_app.py")
print()
print("2. Or use the model in Python:")
print("   from world_development_model import WorldDevelopmentClusteringModel")
print("   model = WorldDevelopmentClusteringModel.load('world_development_kmeans.pkl')")
print("   predictions = model.predict(df)")
print()
