import numpy as np
import pandas as pd
from scipy import stats


url = "https://raw.githubusercontent.com/VincentGranville/Main/main/iot_security.csv"
data = pd.read_csv(url)
data.columns

row_counts = data.groupby(list(data.columns)).size().reset_index(name='multiplicity')
print("\nMultiplicity distribution:\n", row_counts['multiplicity'].value_counts().sort_index())

# Identify the two largest clusters
top2 = row_counts.nlargest(2, 'multiplicity')
print("\nTwo largest clusters (top rows):")
print(top2[['multiplicity'] + list(data.columns)].head())

# Step 1: Split into A and B
# A: top2 clusters + any row with multiplicity >= 4
A_mask = (row_counts['multiplicity'] >= 4) | row_counts['multiplicity'].isin(top2['multiplicity'])
A_unique = row_counts[A_mask].copy()
A_unique['size'] = A_unique['multiplicity']
A_unique = A_unique.drop(columns=['multiplicity'])

# B: all other unique rows (with size)
B_unique = row_counts[~A_mask].copy()
B_unique['size'] = B_unique['multiplicity']
B_unique = B_unique.drop(columns=['multiplicity'])

print(f"\nA: {len(A_unique)} unique rows, total original obs = {A_unique['size'].sum()}")
print(f"B: {len(B_unique)} unique rows, total original obs = {B_unique['size'].sum()}")

# Step 2: Create C & split into C1/C2
# Remove 'src_port' and 'size' from B, keep unique rows, compute multiplicity
cols_for_C = [c for c in B_unique.columns if c not in ['src_port', 'size']]
C_raw = B_unique[cols_for_C].copy()
C_counts = C_raw.groupby(list(C_raw.columns)).size().reset_index(name='multiplicity')

# C1: multiplicity > 1, C2: multiplicity == 1
C1_raw = C_counts[C_counts['multiplicity'] > 1].copy()
C2_raw = C_counts[C_counts['multiplicity'] == 1].copy()

# Outlier removal in C1 (IQR)
if len(C1_raw) > 0:
    cont_cols = [c for c in C1_raw.columns if c != 'multiplicity' and
                 pd.api.types.is_numeric_dtype(C1_raw[c]) and C1_raw[c].nunique() > 10]
    if cont_cols:
        mask = pd.Series(True, index=C1_raw.index)
        for col in cont_cols:
            z = np.abs(stats.zscore(C1_raw[col].dropna()))
            mask &= (z <= 3).reindex(C1_raw.index, fill_value=True)
        removed = C1_raw[~mask]
        # If removal removes more than half, keep all (to avoid empty C1)
        if len(removed) > len(C1_raw) // 2:
            print(f"Warning: IQR/Z-score would remove {len(removed)} rows from C1. Keeping all.")
        else:
            C1_raw = C1_raw[mask].copy()
            print(f"Removed {len(removed)} outliers from C1")
    else:
        print("No continuous columns for outlier removal in C1")


print(f"\nC1: {len(C1_raw)} unique rows, total obs = {C1_raw['multiplicity'].sum()}")
print(f"C2: {len(C2_raw)} unique rows, total obs = {C2_raw['multiplicity'].sum()}")

# Step 3: Reconstruct 'scr port' maps
# Merge C1 and C2 back with B to get original scr port and size
feat_cols = [c for c in C1_raw.columns if c != 'multiplicity']  # same for C2, i.e. common feature cols
B_with_src = B_unique[feat_cols + ['src_port', 'size']]

C1_full = C1_raw.merge(B_with_src, on=feat_cols, how='left')
C2_full = C2_raw.merge(B_with_src, on=feat_cols, how='left')

# Build maps: sum of sizes per scr port
map_C1 = C1_full.groupby('src_port')['size'].sum().reset_index(name='count')
map_C2 = C2_full.groupby('src_port')['size'].sum().reset_index(name='count')

print(f"\nMap C1: {len(map_C1)} distinct ports, total count = {map_C1['count'].sum()}")
print(f"Map C2: {len(map_C2)} distinct ports, total count = {map_C2['count'].sum()}")


# map_C1.to_csv('iot_C1_map.csv', index=False)
# map_C2.to_csv('iot_C2_map.csv', index=False)

# Generate synthetic observations for C2 using NoGAN algorithm
# Synthetic generation (NoGAN placeholder)
