"""
main.py
=======

Main script to use the World Development Clustering Model

This script demonstrates how to import and use the model
from the separate module file.
"""

# Import necessary libraries
import pandas as pd
import numpy as np

# Import the model class from our separate module
from world_development_model import WorldDevelopmentClusteringModel

print("=" * 80)
print("WORLD DEVELOPMENT CLUSTERING - MAIN SCRIPT")
print("=" * 80)
print()

# ============================================================================
# STEP 1: LOAD YOUR DATA
# ============================================================================

print("Step 1: Loading data...")
print("-" * 80)

# Load your Excel data
df = pd.read_excel('World_development_mesurement.xlsx')

print(f"Loaded {len(df)} records")
print(f"Columns: {df.shape[1]}")
print()

# ============================================================================
# STEP 2: LOAD THE TRAINED MODEL
# ============================================================================

print("Step 2: Loading trained model...")
print("-" * 80)

# Load the saved model
model = WorldDevelopmentClusteringModel.load('world_development_kmeans.pkl')
print()

# ============================================================================
# STEP 3: MAKE PREDICTIONS
# ============================================================================

print("Step 3: Making predictions...")
print("-" * 80)

# Get predictions
predictions = model.predict(df)

# Add predictions to dataframe
df['Final_Cluster'] = predictions

print(f"Classified {len(df)} records")
print()

# ============================================================================
# STEP 4: ANALYZE RESULTS
# ============================================================================

print("Step 4: Analyzing cluster distribution...")
print("-" * 80)
print()

# Cluster distribution
print("Cluster Distribution:")
cluster_counts = df['Final_Cluster'].value_counts().sort_index()

for cluster, count in cluster_counts.items():
    percentage = (count / len(df)) * 100
    print(f"  Cluster {cluster}: {count:4d} countries ({percentage:5.1f}%)")
print()

# ============================================================================
# STEP 5: ADD CLUSTER NAMES
# ============================================================================

print("Step 5: Adding cluster names...")
print("-" * 80)

# Define cluster names (based on our analysis)
cluster_names = {
    0: "Developed Countries",
    1: "Developing Countries",
    2: "Emerging Economies"
}

# Add cluster names to dataframe
df['Cluster_Name'] = df['Final_Cluster'].map(cluster_names)

print("Cluster names added")
print()

# ============================================================================
# STEP 6: SHOW SAMPLE RESULTS
# ============================================================================

print("Step 6: Sample results per cluster...")
print("-" * 80)
print()

for cluster in sorted(df['Final_Cluster'].unique()):
    cluster_name = cluster_names[cluster]
    sample_countries = df[df['Final_Cluster'] == cluster]['Country'].head(5).tolist()
    print(f"{cluster_name}:")
    print(f"  {', '.join(sample_countries)}")
    print()

# ============================================================================
# STEP 7: SAVE RESULTS
# ============================================================================

print("Step 7: Saving results...")
print("-" * 80)

# Save to Excel
output_file = 'clustered_results.xlsx'
df[['Country', 'Final_Cluster', 'Cluster_Name']].to_excel(output_file, index=False)

print(f"Results saved to: {output_file}")
print()

# ============================================================================
# DONE!
# ============================================================================

print("=" * 80)
print("COMPLETED SUCCESSFULLY!")
print("=" * 80)
print()
print(f"Total countries classified: {len(df)}")
print(f"Output file: {output_file}")
print()
