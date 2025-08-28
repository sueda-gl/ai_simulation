#!/usr/bin/env python3
"""Quick test of dependent variable resampling mode"""

import sys
sys.path.insert(0, '.')

from src.orchestrator_depvar import OrchestratorDepVar

# Initialize
print("Initializing OrchestratorDepVar...")
orch = OrchestratorDepVar()

# Get stats
stats = orch.get_empirical_stats()
print(f"\nEmpirical stats:")
for k, v in stats.items():
    print(f"  {k}: {v}")

# Get original distribution
orig = orch.get_empirical_distribution()
print(f"\nOriginal distribution shape: {orig.shape}")
print(f"First 10 values: {orig[:10]}")

# Run simulation
print("\nRunning simulation with 1000 agents...")
df = orch.run_simulation(n_agents=1000, seed=42)
print(f"Result shape: {df.shape}")
print(f"Result columns: {df.columns.tolist()}")
print(f"Result stats:")
print(df['donation_default'].describe())

print("\nTest complete!")
