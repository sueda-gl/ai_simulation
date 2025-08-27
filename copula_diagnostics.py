#!/usr/bin/env python3
"""
Gaussian Copula Diagnostics for AI Agent Simulation Framework
Based on the comprehensive workflow provided by the user.
"""

import numpy as np
import pandas as pd
from scipy import stats
from numpy.random import default_rng
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from pathlib import Path
import sys

# Add project root to path
sys.path.append('.')
from src.build_master_traits import get_master_trait_list
from src.validate_traits import merged
from src.trait_engine import TraitEngine
from src.orchestrator import Orchestrator

print("🔬 GAUSSIAN COPULA DIAGNOSTICS")
print("=" * 80)

# ============================================================================
# 0. Setup - Get the "pseudo-observations"
# ============================================================================
print("\n📊 SETUP: Loading data and computing pseudo-observations...")

# Get traits and clean data
traits = get_master_trait_list()
print(f"Traits: {traits}")

# Load and clean data (same as copula training)
df = merged[traits].copy().dropna()
print(f"Sample size: {len(df)} participants")

# Handle categorical variables (same encoding as correlation analysis)
df_encoded = df.copy()

# Encode Group_experiment as numeric (ordered by subsidy level)
group_map = {'NoSub': 0, 'MidSub': 1, 'HighSub': 2}
df_encoded['Group_experiment'] = df['Group_experiment'].map(group_map)

# Encode Study Program (alphabetical order)
study_programs = sorted(df['Study Program'].unique())
study_map = {prog: i for i, prog in enumerate(study_programs)}
df_encoded['Study Program'] = df['Study Program'].map(study_map)

X = df_encoded[traits]
print("Data types after encoding:")
print(X.dtypes)

# Convert to pseudo-observations (percentile ranks)
U = X.rank(method="average") / (len(X) + 1)
print(f"U shape: {U.shape}, range: [{U.min().min():.4f}, {U.max().max():.4f}]")

# Map to standard normals
Z = pd.DataFrame(stats.norm.ppf(U), columns=traits)
print(f"Z shape: {Z.shape}, no inf/nan: {np.isfinite(Z.values).all()}")

# ============================================================================
# 1. Estimate Σ and compare with fitted model
# ============================================================================
print("\n📐 STEP 1: Correlation matrix estimation...")

Sigma_empirical = Z.corr().values
print("Empirical correlation matrix (from raw data):")
print(pd.DataFrame(Sigma_empirical, columns=traits, index=traits).round(3))

# Load the actual fitted copula model
model_path = Path('config/trait_model.pkl')
if model_path.exists():
    model = joblib.load(model_path)
    Sigma_fitted = model['Sigma']
    print("\nFitted copula correlation matrix:")
    print(pd.DataFrame(Sigma_fitted, columns=traits, index=traits).round(3))
    
    # Compare fitted vs empirical
    diff_matrix = Sigma_fitted - Sigma_empirical
    print(f"\nMax absolute difference (fitted - empirical): {np.abs(diff_matrix).max():.6f}")
    print("Difference matrix:")
    print(pd.DataFrame(diff_matrix, columns=traits, index=traits).round(4))
else:
    print("❌ Fitted copula model not found!")
    Sigma_fitted = Sigma_empirical
    print("Using empirical correlation matrix for diagnostics.")

Sigma_hat = Sigma_fitted
p = len(traits)

# Check positive definiteness
eigenvals = np.linalg.eigvals(Sigma_hat)
print(f"Eigenvalues: {eigenvals}")
print(f"Matrix is positive definite: {np.all(eigenvals > 0)}")
print(f"Condition number: {np.linalg.cond(Sigma_hat):.2f}")

# ============================================================================
# 2. Diagnostic A - Bootstrap stability of Σ
# ============================================================================
print("\n🎲 DIAGNOSTIC A: Bootstrap stability of correlation matrix...")

rng = default_rng(42)
B = 500
sigmas = np.empty((B, p, p))

print(f"Running {B} bootstrap samples...")
for b in range(B):
    if (b + 1) % 100 == 0:
        print(f"  Bootstrap {b+1}/{B}")
    
    boot_indices = rng.choice(len(Z), size=len(Z), replace=True)
    boot_sample = Z.iloc[boot_indices]
    sigmas[b] = boot_sample.corr().values

sigma_std = sigmas.std(axis=0)
sigma_ci = np.quantile(sigmas, [0.025, 0.975], axis=0)

print(f"Max bootstrap SD: {sigma_std.max():.4f}")
print(f"Max off-diagonal bootstrap SD: {sigma_std[~np.eye(p, dtype=bool)].max():.4f}")

# Show most unstable correlations
off_diag_mask = ~np.eye(p, dtype=bool)
max_std_idx = np.unravel_index(sigma_std[off_diag_mask].argmax(), sigma_std.shape)
print(f"Most unstable correlation: {traits[max_std_idx[0]]} <-> {traits[max_std_idx[1]]}")
print(f"  SD = {sigma_std[max_std_idx]:.4f}")

# Rule of thumb check
stable_correlations = (sigma_std[off_diag_mask] < 0.05).mean()
print(f"Fraction of stable correlations (SD < 0.05): {stable_correlations:.1%}")

if sigma_std[off_diag_mask].max() < 0.05:
    print("✅ PASS: Correlation matrix is bootstrap-stable")
else:
    print("⚠️  CAUTION: Some correlations show high bootstrap variability")

# ============================================================================
# 3. Diagnostic B - Match rank correlations (Kendall's tau)
# ============================================================================
print("\n🔗 DIAGNOSTIC B: Rank correlation consistency...")

tau_empirical = Z.corr(method="kendall").values
tau_model = (2/np.pi) * np.arcsin(Sigma_hat)
tau_diff = np.abs(tau_empirical - tau_model)

print(f"Max |τ_empirical - τ_model|: {tau_diff[off_diag_mask].max():.4f}")
print(f"Mean |τ_empirical - τ_model|: {tau_diff[off_diag_mask].mean():.4f}")

# Show largest discrepancies
max_diff_idx = np.unravel_index(tau_diff[off_diag_mask].argmax(), tau_diff.shape)
print(f"Largest discrepancy: {traits[max_diff_idx[0]]} <-> {traits[max_diff_idx[1]]}")
print(f"  τ_empirical = {tau_empirical[max_diff_idx]:.4f}")
print(f"  τ_model = {tau_model[max_diff_idx]:.4f}")
print(f"  |difference| = {tau_diff[max_diff_idx]:.4f}")

if tau_diff[off_diag_mask].max() < 0.05:
    print("✅ PASS: Rank correlations match well")
else:
    print("⚠️  CAUTION: Some rank correlations show discrepancies")

# ============================================================================
# 4. Diagnostic C - Empirical vs Gaussian copula CDF
# ============================================================================
print("\n📈 DIAGNOSTIC C: Copula CDF comparison...")

def empirical_copula(U_data, n_grid=20):
    """Compute empirical copula CDF on a grid"""
    grid = np.linspace(0, 1, n_grid)
    c = np.zeros((n_grid, n_grid))
    for i, u in enumerate(grid):
        for j, v in enumerate(grid):
            c[i, j] = ((U_data.iloc[:, 0] <= u) & (U_data.iloc[:, 1] <= v)).mean()
    return grid, c

# Focus on the most correlated pair for visualization
corr_abs = np.abs(Sigma_hat)
np.fill_diagonal(corr_abs, 0)  # Ignore diagonal
max_corr_idx = np.unravel_index(corr_abs.argmax(), corr_abs.shape)
trait1, trait2 = traits[max_corr_idx[0]], traits[max_corr_idx[1]]

print(f"Analyzing strongest correlation: {trait1} <-> {trait2}")
print(f"Correlation: {Sigma_hat[max_corr_idx]:.4f}")

# Empirical copula
u_grid, C_empirical = empirical_copula(U[[trait1, trait2]])

# Simulate from Gaussian copula
L = np.linalg.cholesky(Sigma_hat)
Z_sim = rng.standard_normal((len(Z), p)) @ L.T
U_sim = stats.norm.cdf(Z_sim)
U_sim_df = pd.DataFrame(U_sim, columns=traits)
_, C_gaussian = empirical_copula(U_sim_df[[trait1, trait2]])

# Compare
copula_diff = np.abs(C_empirical - C_gaussian)
max_copula_diff = copula_diff.max()
mean_copula_diff = copula_diff.mean()

print(f"Max |C_empirical - C_gaussian|: {max_copula_diff:.4f}")
print(f"Mean |C_empirical - C_gaussian|: {mean_copula_diff:.4f}")

if max_copula_diff < 0.1:
    print("✅ PASS: Copula CDFs match well")
else:
    print("⚠️  CAUTION: Copula CDFs show discrepancies")

# ============================================================================
# 5. Diagnostic D - Tail dependence
# ============================================================================
print("\n🎯 DIAGNOSTIC D: Tail dependence analysis...")

alpha = 0.95
n_pairs = 0
total_upper_emp = 0
total_upper_sim = 0

print(f"Checking upper tail dependence (α = {alpha})...")

for i in range(p):
    for j in range(i+1, p):
        # Empirical upper tail dependence
        upper_tail_emp = ((U.iloc[:, i] > alpha) & (U.iloc[:, j] > alpha)).mean() / (1 - alpha)
        upper_tail_sim = ((U_sim[:, i] > alpha) & (U_sim[:, j] > alpha)).mean() / (1 - alpha)
        
        total_upper_emp += upper_tail_emp
        total_upper_sim += upper_tail_sim
        n_pairs += 1
        
        if upper_tail_emp > 0.05:  # Only report significant tail dependence
            print(f"  {traits[i]} <-> {traits[j]}:")
            print(f"    Empirical λ_U: {upper_tail_emp:.4f}")
            print(f"    Gaussian λ_U:  {upper_tail_sim:.4f}")

avg_upper_emp = total_upper_emp / n_pairs
avg_upper_sim = total_upper_sim / n_pairs

print(f"Average empirical upper tail λ_U: {avg_upper_emp:.4f}")
print(f"Average Gaussian λ_U (should be ~0): {avg_upper_sim:.4f}")

if avg_upper_emp < 0.05:
    print("✅ PASS: No significant tail dependence detected")
else:
    print("⚠️  CAUTION: Significant tail dependence present - consider t-copula")

# ============================================================================
# 6. Diagnostic E - Outcome sensitivity
# ============================================================================
print("\n🎮 DIAGNOSTIC E: Decision outcome sensitivity...")

try:
    # Test with different correlation structures
    n_test = 1000
    test_seed = 42
    
    print(f"Testing with {n_test} agents, seed = {test_seed}")
    
    # Original Gaussian copula
    engine_gaussian = TraitEngine()
    orchestrator = Orchestrator()
    
    print("Running Gaussian copula simulation...")
    df_gaussian = orchestrator.run_simulation(n_test, test_seed, 'donation_default')
    mean_gaussian = df_gaussian['donation_default'].mean()
    
    # Create independent version by modifying the correlation matrix
    model_independent = model.copy()
    model_independent['Sigma'] = np.eye(p)  # Identity matrix = independence
    
    # Save temporary independent model
    temp_model_path = Path('config/trait_model_independent.pkl')
    joblib.dump(model_independent, temp_model_path)
    
    # Test independent version (would need to reload TraitEngine)
    print("Note: Independent comparison would require reloading TraitEngine")
    print("This is a limitation of the current architecture")
    
    # Bootstrap comparison using actual data rows
    print("Running bootstrap simulation...")
    boot_indices = rng.choice(len(df), size=n_test, replace=True)
    df_bootstrap = df.iloc[boot_indices].reset_index(drop=True)
    
    # Run decision on bootstrap sample (simplified)
    # This would require running the decision module directly
    print("Bootstrap comparison would require direct decision module execution")
    
    print(f"Gaussian copula mean donation rate: {mean_gaussian:.4f}")
    
    # Cleanup
    if temp_model_path.exists():
        temp_model_path.unlink()
        
except Exception as e:
    print(f"❌ Outcome sensitivity test failed: {e}")
    print("This test requires a more flexible architecture for comparison")

# ============================================================================
# Summary Report
# ============================================================================
print("\n" + "=" * 80)
print("📋 DIAGNOSTIC SUMMARY REPORT")
print("=" * 80)

tests_passed = 0
total_tests = 5

print("1. Bootstrap Stability:")
if sigma_std[off_diag_mask].max() < 0.05:
    print("   ✅ PASS - Correlation matrix is stable")
    tests_passed += 1
else:
    print("   ⚠️  CAUTION - High bootstrap variability")

print("2. Rank Correlation Consistency:")
if tau_diff[off_diag_mask].max() < 0.05:
    print("   ✅ PASS - Kendall's tau matches well")
    tests_passed += 1
else:
    print("   ⚠️  CAUTION - Rank correlation discrepancies")

print("3. Copula CDF Match:")
if max_copula_diff < 0.1:
    print("   ✅ PASS - Empirical and Gaussian CDFs align")
    tests_passed += 1
else:
    print("   ⚠️  CAUTION - CDF discrepancies detected")

print("4. Tail Dependence:")
if avg_upper_emp < 0.05:
    print("   ✅ PASS - No excessive tail dependence")
    tests_passed += 1
else:
    print("   ⚠️  CAUTION - Significant tail dependence")

print("5. Outcome Sensitivity:")
print("   ⏸️  PARTIAL - Architecture limitations prevent full test")

print(f"\nOverall Score: {tests_passed}/{total_tests-1} tests passed")

if tests_passed >= 3:
    print("🎉 CONCLUSION: Gaussian copula is adequate for your simulation!")
else:
    print("🤔 CONCLUSION: Consider alternative copula models")

print("\n📊 Key Statistics:")
print(f"  - Sample size: {len(df)} participants")
print(f"  - Traits: {len(traits)}")
print(f"  - Max correlation: {np.abs(Sigma_hat[off_diag_mask]).max():.4f}")
print(f"  - Condition number: {np.linalg.cond(Sigma_hat):.2f}")
print(f"  - Ridge regularization: λ = 0.1 (from training)")

print("\n" + "=" * 80)
print("Diagnostics complete! 🔬")