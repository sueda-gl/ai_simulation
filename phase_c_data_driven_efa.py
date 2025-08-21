#!/usr/bin/env python3
"""
Phase C (data-driven): Discover natural variable clusters through EFA
====================================================================

Instead of pre-defining conceptual bundles, this approach:
1. Runs EFA on ALL survey variables + DV
2. Groups variables by their primary factor loadings
3. Re-runs EFA within each discovered cluster
4. Compares results to theory-based approach
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from factor_analyzer import FactorAnalyzer, calculate_kmo, calculate_bartlett_sphericity

def main():
    # Load data
    df = pd.read_csv("analysis_table.csv")
    
    # Get all survey variables + DV
    survey_vars = [c for c in df.columns if c.endswith("_std") and c != "Prosocial_DV"]
    all_vars = ["Prosocial_DV"] + survey_vars
    
    # Standardize everything for EFA
    X_raw = df[all_vars].dropna()
    X = (X_raw - X_raw.mean()) / X_raw.std(ddof=0)
    
    print(f"Running initial EFA on {len(all_vars)} variables...")
    
    # Step 1: Determine number of factors
    fa_test = FactorAnalyzer(rotation=None)
    fa_test.fit(X)
    eigenvalues, _ = fa_test.get_eigenvalues()
    n_factors = (eigenvalues > 1).sum()
    print(f"Eigenvalue > 1 rule suggests {n_factors} factors")
    
    # Step 2: Run EFA with oblique rotation
    fa = FactorAnalyzer(n_factors=n_factors, rotation='oblimin', method='minres')
    fa.fit(X)
    
    # Step 3: Create loadings matrix
    loadings = pd.DataFrame(
        fa.loadings_, 
        index=all_vars, 
        columns=[f"Factor{i+1}" for i in range(n_factors)]
    )
    
    print("\nFactor loadings matrix:")
    print(loadings.round(3))
    
    # Step 4: Assign variables to primary factors (threshold = 0.4)
    threshold = 0.4
    discovered_clusters = {}
    
    for factor in loadings.columns:
        # Find variables that load > threshold on this factor
        high_loaders = loadings[loadings[factor].abs() > threshold].index.tolist()
        if len(high_loaders) >= 2:  # Need at least 2 variables
            discovered_clusters[factor] = high_loaders
            print(f"\n{factor} ({len(high_loaders)} variables):")
            for var in high_loaders:
                loading = loadings.loc[var, factor]
                print(f"  {var}: {loading:.3f}")
    
    # Step 5: Check quality of discovered clusters
    print("\n" + "="*50)
    print("CLUSTER QUALITY CHECK")
    print("="*50)
    
    valid_clusters = {}
    for cluster_name, variables in discovered_clusters.items():
        if len(variables) >= 3:  # Need minimum for EFA
            cluster_data = df[variables].dropna()
            cluster_std = (cluster_data - cluster_data.mean()) / cluster_data.std(ddof=0)
            
            # KMO test
            kmo_all, kmo_overall = calculate_kmo(cluster_std)
            chi2, p_val = calculate_bartlett_sphericity(cluster_std)
            
            print(f"\n{cluster_name}:")
            print(f"  Variables: {[v.replace('_std', '') for v in variables]}")
            print(f"  KMO: {kmo_overall:.3f}")
            print(f"  Bartlett p: {p_val:.4g}")
            
            if kmo_overall >= 0.6:
                valid_clusters[cluster_name] = variables
                print(f"  ✅ VALID - good for EFA")
            else:
                print(f"  ❌ POOR - KMO too low")
        else:
            print(f"\n{cluster_name}: Only {len(variables)} variables - too few for EFA")
    
    # Step 6: Run final EFA on valid clusters
    print("\n" + "="*50)
    print("FINAL CLUSTER EFAS")
    print("="*50)
    
    all_factor_scores = []
    
    for cluster_name, variables in valid_clusters.items():
        print(f"\n{cluster_name}:")
        
        # Add DV to each cluster for EFA (crucial for behavioral prediction)
        cluster_vars_with_dv = ["Prosocial_DV"] + variables
        cluster_data = df[cluster_vars_with_dv].dropna()
        cluster_std = (cluster_data - cluster_data.mean()) / cluster_data.std(ddof=0)
        
        # Determine factors for this cluster
        fa_test = FactorAnalyzer(rotation=None)
        fa_test.fit(cluster_std)
        eigenvalues, _ = fa_test.get_eigenvalues()
        n_factors_cluster = min(2, (eigenvalues > 1).sum()) or 1  # Max 2 factors per cluster
        
        # Run EFA
        fa_cluster = FactorAnalyzer(n_factors=n_factors_cluster, rotation='oblimin', method='minres')
        fa_cluster.fit(cluster_std)
        
        # Generate factor scores
        scores = fa_cluster.transform(cluster_std)
        score_cols = [f"{cluster_name}_F{i+1}_score" for i in range(n_factors_cluster)]
        score_df = pd.DataFrame(scores, index=cluster_std.index, columns=score_cols)
        
        # Standardize factor scores
        score_df = (score_df - score_df.mean()) / score_df.std(ddof=0)
        all_factor_scores.append(score_df)
        
        print(f"  Factors extracted: {n_factors_cluster} (including DV)")
        print(f"  Variables in EFA: {[v.replace('_std', '') for v in cluster_vars_with_dv]}")
        print(f"  Factor scores: {score_cols}")
    
    # Step 7: Combine all factor scores
    if all_factor_scores:
        final_scores = pd.concat(all_factor_scores, axis=1)
        
        # Merge with original data
        merged = pd.concat([df, final_scores], axis=1)
        merged.to_csv("analysis_table_data_driven_factors.csv", index=False)
        
        print(f"\n✅ Generated {final_scores.shape[1]} data-driven factor scores")
        print("Saved to: analysis_table_data_driven_factors.csv")
        
        # Save loadings
        loadings.to_csv("data_driven_factor_loadings.csv")
        print("Initial loadings saved to: data_driven_factor_loadings.csv")
    
    return valid_clusters, loadings

if __name__ == "__main__":
    clusters, loadings = main() 