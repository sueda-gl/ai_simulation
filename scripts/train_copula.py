# scripts/train_copula.py
import joblib, numpy as np, pandas as pd
from scipy.stats import norm
from pathlib import Path
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.build_master_traits import get_master_trait_list
from src.validate_traits import merged    # uses the merge already done

MODEL_OUT = Path(__file__).resolve().parents[1] / "config" / "trait_model.pkl"
traits    = get_master_trait_list()
df        = merged[traits].copy()

print(f"Training copula on {len(traits)} traits from {len(df)} participants...")

# Check for missing values
missing_counts = df.isnull().sum()
if missing_counts.any():
    print("Missing values detected:")
    for col, count in missing_counts.items():
        if count > 0:
            print(f"  {col}: {count} missing")
    print("Dropping rows with any missing values...")
    df = df.dropna()
    print(f"Remaining participants after dropping missing: {len(df)}")

# --- 1. rank-transform to uniforms (avoid exact 0 and 1)
def to_uniform(col):
    ranks = col.rank(method="average", pct=True).values
    # Clip to avoid exact 0 and 1 which map to -inf and +inf
    eps = 1e-6
    return np.clip(ranks, eps, 1-eps)

U = np.column_stack([to_uniform(df[c]) for c in traits])

# --- 2. map to latent normals
Z = norm.ppf(U)

# Check for any remaining NaN/inf values
if np.any(~np.isfinite(Z)):
    print("Warning: Non-finite values in latent normal matrix")
    print("This usually means extreme uniform values (0 or 1)")

# --- 3. estimate correlation Σ with ridge
Sigma = np.corrcoef(Z, rowvar=False)
lam   = 0.1  # Increased ridge regularization
Sigma = (1-lam)*Sigma + lam*np.eye(len(traits))

# Ensure positive definiteness
eigenvals = np.linalg.eigvals(Sigma)
min_eigenval = np.min(eigenvals)
if min_eigenval <= 0:
    print(f"Warning: Minimum eigenvalue {min_eigenval:.6f}, adding more regularization")
    lam = 0.2
    Sigma = (1-lam)*np.corrcoef(Z, rowvar=False) + lam*np.eye(len(traits))

print(f"Correlation matrix shape: {Sigma.shape}")
print(f"Ridge regularization: λ = {lam}")

# --- 4. build inverse-CDF decoders (store raw data for reconstruction)
decoders = {}
for j, col in enumerate(traits):
    values, quant = np.unique(df[col], return_counts=True)
    cdf = np.cumsum(quant)/quant.sum()
    # Store the raw data instead of lambda functions for pickle compatibility
    decoders[col] = {
        'values': values,
        'cdf': cdf,
        'dtype': df[col].dtype
    }

print(f"Built {len(decoders)} inverse-CDF decoders")

# --- 5. serialize to disk
joblib.dump({"Sigma": Sigma, "decoders": decoders, "traits": traits}, MODEL_OUT)
print("✅  Copula model saved to", MODEL_OUT)