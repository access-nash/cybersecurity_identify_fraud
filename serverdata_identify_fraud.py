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
print("\nTwo largest clusters :")
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
C_counts = B_unique.groupby(cols_for_C)['size'].sum().reset_index(name='multiplicity')

# C1: multiplicity > 1, C2: multiplicity == 1
C1_raw = C_counts[C_counts['multiplicity'] > 1].copy()
C2_raw = C_counts[C_counts['multiplicity'] == 1].copy()

# Outlier removal in C1 (z score)
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
            print(f"Z-score would remove {len(removed)} rows from C1. Keeping all.")
        else:
            C1_raw = C1_raw[mask].copy()
            print(f"Removed {len(removed)} outliers from C1")


print(f"\nC1: {len(C1_raw)} unique rows, total obs = {C1_raw['multiplicity'].sum()}")
print(f"C2: {len(C2_raw)} unique rows, total obs = {C2_raw['multiplicity'].sum()}")

# Step 3: Reconstruct 'src port' maps
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

# Step 4: Generate synthetic observations for C2 using NoGAN algorithm

# Synthetic data generation (NoGAN taken from earlier project in Vincent's book)
def generate_synthetic_C2(n_samples, C2, map_C2):
    """
    Generate synthetic observations for C2 using NoGAN.
    """
    if n_samples == 0 or len(C2) == 0:
        return pd.DataFrame()

    # features for NoGAN
    features = ['bidirectional_syn_packets', 'src2dst_syn_packets', 'application_category_name', 'application_confidence',
                'src2dst_mean_ps', 'src2dst_psh_packets', 'bidirectional_mean_ps', 'label']

    data_training = C2[features].copy()

    if 'application_category_name' in data_training.columns:
        data_training['application_category_name'], _ = pd.factorize(data_training['application_category_name'])

    np.random.seed(105)

    # Hyperparameters: number of bins per feature
    unique_cat = data_training[
        'application_category_name'].nunique() if 'application_category_name' in data_training.columns else 5
    bins_per_feature = [40, 40, max(unique_cat, 2), 40, 40, 40, 40, 4]
    bins_per_feature = np.array(bins_per_feature).astype(int)

    n_features = len(features)
    nobs = len(data_training)
    eps = 1e-10

    # Create quantile table
    pc_table2 = []
    for k in range(n_features):
        label = features[k]
        incr = 1 / bins_per_feature[k]
        pc = np.arange(0, 1 + eps, incr)
        arr = np.quantile(data_training[label], pc, axis=0)
        pc_table2.append(arr)

    # Bin each observation
    npdata = data_training.to_numpy()
    bin_count = {}
    bin_obs = {}
    for obs in npdata:
        key = []
        for k in range(n_features):
            idx = 0
            arr = pc_table2[k]
            while obs[k] >= arr[idx] and idx < bins_per_feature[k]:
                idx += 1
            idx = idx - 1
            key.append(idx)
        skey = str(key)
        if skey in bin_count:
            bin_count[skey] += 1
            bin_obs[skey] += "~" + str(obs.tolist())
        else:
            bin_count[skey] = 1
            bin_obs[skey] = str(obs.tolist())

    # Generate synthetic observations
    def random_bin_counts(n, bin_count):
        pvals = [bin_count[skey] / nobs for skey in bin_count]
        return np.random.multinomial(n, pvals)

    nobs_synth = n_samples
    bin_count_random = random_bin_counts(nobs_synth, bin_count)
    ikey = 0

    data_synth = []

    for skey in bin_count:
        count = bin_count_random[ikey]
        ikey += 1
        key = eval(skey)

        L_bounds = []
        U_bounds = []
        for k in range(n_features):
            arr = pc_table2[k]
            L_bounds.append(arr[key[k]])
            U_bounds.append(arr[1 + key[k]])

        for _ in range(count):
            new_obs = np.empty(n_features)
            for k in range(n_features):
                new_obs[k] = np.random.uniform(L_bounds[k], U_bounds[k])
            data_synth.append(new_obs)

    synth_df = pd.DataFrame(data_synth, columns=features)

    synth_df['label'] = synth_df['label'].round().astype(int)
    synth_df['application_category_name'] = synth_df['application_category_name'].round().astype(int)

    # Add src_port from map_C2
    ports = map_C2['src_port'].values
    probs = map_C2['count'] / map_C2['count'].sum()
    synth_df['src_port'] = np.random.choice(ports, size=n_samples, p=probs)


    return synth_df


# Step 5: Compute proportions and multinomial draw for total N = 100000

total_orig = len(data)
n_A_orig = A_unique['size'].sum()
n_C1_orig = C1_raw['multiplicity'].sum() if len(C1_raw) > 0 else 0
n_C2_orig = C2_raw['multiplicity'].sum() if len(C2_raw) > 0 else 0

p1 = n_A_orig / total_orig
p2 = n_C1_orig / total_orig
p3 = n_C2_orig / total_orig
print(f"\nProportions: p1={p1:.4f}, p2={p2:.4f}, p3={p3:.4f}")

N = 100000
np.random.seed(105)
n1, n2, n3 = np.random.multinomial(N, [p1, p2, p3])
print(f"Multinomial counts: n1={n1}, n2={n2}, n3={n3}")


# A: resample with weights=size
synth_A = A_unique.sample(n=n1, replace=True, weights='size', axis=0).drop(columns=['size']).reset_index(drop=True) if n1 > 0 else pd.DataFrame()

# C1: resample with weights=multiplicity
synth_C1 = C1_raw.sample(n=n2, replace=True, weights='multiplicity', axis=0).drop(columns=['multiplicity']).reset_index(drop=True) if n2 > 0 else pd.DataFrame()

# C2: NoGAN
synth_C2 = generate_synthetic_C2(n3, C2_raw, map_C2)

print(f"\nSynthetic A: {len(synth_A)} rows")
print(f"Synthetic C1: {len(synth_C1)} rows")
print(f"Synthetic C2: {len(synth_C2)} rows")
print(f"Total synthetic: {len(synth_A)+len(synth_C1)+len(synth_C2)}")


full_synth = pd.concat([synth_A, synth_C1, synth_C2], ignore_index=True)
full_synth.to_csv('full_synthetic.csv', index=False)

