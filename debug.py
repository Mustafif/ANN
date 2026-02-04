import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Load your data
dataset = "datasets/HN_100K.csv"
data = pd.read_csv(dataset)

print("="*60)
print("DATA DIAGNOSTICS - Investigating High Validation Loss")
print("="*60)

# 1. Check target variable (V) distribution
print("\n1. TARGET VARIABLE (V) ANALYSIS:")
print(f"   Mean: {data['V'].mean():.6f}")
print(f"   Std: {data['V'].std():.6f}")
print(f"   Min: {data['V'].min():.6f}")
print(f"   Max: {data['V'].max():.6f}")
print(f"   Median: {data['V'].median():.6f}")
print(f"   Range: {data['V'].max() - data['V'].min():.6f}")

# Check for outliers
q1 = data['V'].quantile(0.25)
q3 = data['V'].quantile(0.75)
iqr = q3 - q1
outliers = data[(data['V'] < q1 - 1.5*iqr) | (data['V'] > q3 + 1.5*iqr)]
print(f"   Outliers (IQR method): {len(outliers)} ({len(outliers)/len(data)*100:.2f}%)")

# 2. Check for data quality issues
print("\n2. DATA QUALITY CHECKS:")
print(f"   Total samples: {len(data)}")
print(f"   Samples with V > 1e-8: {len(data[data['V'] > 1e-8])}")
print(f"   Missing values: {data.isnull().sum().sum()}")

# 3. Check feature distributions
print("\n3. FEATURE STATISTICS:")
features = ["S0", "m", "r", "T", "callput", "alpha", "beta", "omega", "gamma", "lambda"]
for feat in features:
    if feat in data.columns:
        print(f"   {feat:10s}: mean={data[feat].mean():10.4f}, std={data[feat].std():10.4f}, "
              f"min={data[feat].min():10.4f}, max={data[feat].max():10.4f}")

# 4. Check for extreme values in log-transformed features
print("\n4. LOG-TRANSFORMED FEATURES CHECK:")
log_features = ["alpha", "beta", "omega", "gamma", "lambda"]
for feat in log_features:
    if feat in data.columns:
        log_vals = np.log(data[feat].values + 1e-8)
        print(f"   log({feat}): mean={log_vals.mean():10.4f}, std={log_vals.std():10.4f}, "
              f"min={log_vals.min():10.4f}, max={log_vals.max():10.4f}")
        # Check for -inf or very large negative values
        neg_inf = np.sum(np.isinf(log_vals))
        very_negative = np.sum(log_vals < -20)
        if neg_inf > 0:
            print(f"      WARNING: {neg_inf} -inf values!")
        if very_negative > 0:
            print(f"      WARNING: {very_negative} values < -20!")

# 5. Scale analysis
print("\n5. SCALE MISMATCH ANALYSIS:")
print("   If features have vastly different scales, this can cause issues:")
all_features = []
for feat in features:
    if feat in data.columns:
        all_features.append(data[feat].values)

# Add log features
for feat in log_features:
    if feat in data.columns:
        all_features.append(np.log(data[feat].values + 1e-8))

feature_scales = [np.std(f) for f in all_features]
print(f"   Min feature std: {min(feature_scales):.6f}")
print(f"   Max feature std: {max(feature_scales):.6f}")
print(f"   Ratio (max/min): {max(feature_scales)/min(feature_scales):.2f}")
if max(feature_scales)/min(feature_scales) > 100:
    print("   ⚠️  WARNING: Large scale differences detected! Consider normalization.")

# 6. Loss function analysis
print("\n6. HUBER LOSS ANALYSIS:")
print("   Huber loss with delta=1.0 (PyTorch default):")
print("   - For |error| <= 1: loss = 0.5 * error²")
print("   - For |error| > 1:  loss = |error| - 0.5")
print("\n   If your predictions are off by >1, loss will be linear with error.")
print("   With V range:", data['V'].min(), "to", data['V'].max())

# Simulate what loss >1 means
print("\n   What does validation loss > 1 mean?")
example_errors = [0.5, 1.0, 1.5, 2.0, 3.0]
for err in example_errors:
    if err <= 1:
        loss = 0.5 * err**2
    else:
        loss = err - 0.5
    print(f"   - Average |prediction - true| = {err:.1f} → Huber loss ≈ {loss:.3f}")

# 7. Visualizations
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Plot 1: Target distribution
axes[0, 0].hist(data['V'], bins=50, edgecolor='black')
axes[0, 0].set_xlabel('V')
axes[0, 0].set_ylabel('Frequency')
axes[0, 0].set_title('Target Variable (V) Distribution')
axes[0, 0].grid(True, alpha=0.3)

# Plot 2: Log scale
axes[0, 1].hist(np.log(data['V'] + 1e-8), bins=50, edgecolor='black')
axes[0, 1].set_xlabel('log(V)')
axes[0, 1].set_ylabel('Frequency')
axes[0, 1].set_title('Log-Transformed Target Distribution')
axes[0, 1].grid(True, alpha=0.3)

# Plot 3: Feature correlations with target
correlations = []
feature_names = []
for feat in features:
    if feat in data.columns:
        corr = data[feat].corr(data['V'])
        correlations.append(corr)
        feature_names.append(feat)

axes[1, 0].barh(feature_names, correlations)
axes[1, 0].set_xlabel('Correlation with V')
axes[1, 0].set_title('Feature Correlations with Target')
axes[1, 0].grid(True, alpha=0.3)

# Plot 4: Box plot of V
axes[1, 1].boxplot(data['V'])
axes[1, 1].set_ylabel('V')
axes[1, 1].set_title('Target Variable Box Plot')
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('data_diagnostics.png')
print("\n📊 Diagnostic plots saved to 'data_diagnostics.png'")
plt.show()

# 8. Recommendations
print("\n" + "="*60)
print("POTENTIAL CAUSES & SOLUTIONS:")
print("="*60)

issues = []
solutions = []

if data['V'].std() > 1:
    issues.append("High variance in target variable")
    solutions.append("Consider normalizing/standardizing the target (V)")

if max(feature_scales)/min(feature_scales) > 100:
    issues.append("Large scale differences between features")
    solutions.append("Normalize/standardize input features")

if len(outliers) > len(data) * 0.05:
    issues.append(f"Many outliers in target ({len(outliers)/len(data)*100:.1f}%)")
    solutions.append("Consider robust scaling or outlier removal")

if data['V'].max() > 10:
    issues.append("Target values are large (max > 10)")
    solutions.append("Use target normalization: (V - mean) / std")

# Check for very small values leading to extreme logs
for feat in log_features:
    if feat in data.columns:
        if (data[feat] < 1e-6).sum() > 0:
            issues.append(f"Very small values in {feat} causing extreme log values")
            solutions.append(f"Increase epsilon in log transform or use different transform")

if not issues:
    issues.append("No obvious data issues detected")
    solutions.append("Check model architecture, learning rate, or training procedure")

print("\nIdentified Issues:")
for i, issue in enumerate(issues, 1):
    print(f"{i}. {issue}")

print("\nRecommended Solutions:")
for i, solution in enumerate(solutions, 1):
    print(f"{i}. {solution}")

print("\n" + "="*60)
