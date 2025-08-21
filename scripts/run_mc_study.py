# scripts/run_mc_study.py
import argparse
import subprocess
import pandas as pd
import numpy as np
from pathlib import Path
import sys
from datetime import datetime
import json

def main():
    parser = argparse.ArgumentParser(description='Run Monte-Carlo study with multiple simulation runs')
    parser.add_argument('--agents', type=int, default=10000,
                       help='Number of synthetic agents per run (default: 10000)')
    parser.add_argument('--runs', type=int, default=500,
                       help='Number of Monte-Carlo repetitions (default: 500)')
    parser.add_argument('--base-seed', type=int, default=1,
                       help='Base random seed (subsequent runs use base_seed + i) (default: 1)')
    parser.add_argument('--decision', type=str, default=None,
                       help='Run only specific decision (default: run all)')
    parser.add_argument('--output-dir', type=str, default='outputs',
                       help='Output directory (default: outputs)')
    parser.add_argument('--keep-individual', action='store_true',
                       help='Keep individual run files (default: delete after aggregation)')
    parser.add_argument('--summary-only', action='store_true',
                       help='Only save summary statistics, not individual runs')
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Initialize results storage
    mc_results = []
    individual_files = []
    
    print(f"Starting Monte-Carlo study with {args.runs} repetitions")
    print(f"Population size: {args.agents} agents per run")
    print(f"Base seed: {args.base_seed}")
    if args.decision:
        print(f"Decision: {args.decision}")
    else:
        print("Running all 13 decisions")
    print("-" * 50)
    
    # Run Monte-Carlo loop
    for i in range(args.runs):
        current_seed = args.base_seed + i
        
        # Build command for individual simulation
        cmd = [
            sys.executable, 'scripts/run_simulation.py',
            '--agents', str(args.agents),
            '--seed', str(current_seed),
            '--format', 'parquet'
        ]
        
        if args.decision:
            cmd.extend(['--decision', args.decision])
        
        # Run simulation
        try:
            print(f"Run {i+1:3d}/{args.runs}: seed={current_seed}", end=" ... ", flush=True)
            
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            
            # Parse output to find result file
            output_lines = result.stdout.strip().split('\n')
            result_file = None
            for line in output_lines:
                if 'Results saved to:' in line:
                    result_file = line.split('Results saved to:')[1].strip()
                    break
            
            if not result_file or not Path(result_file).exists():
                print("ERROR: Could not find output file")
                continue
            
            # Load results and extract key statistics
            df = pd.read_parquet(result_file)
            
            # Calculate summary statistics for this run
            run_stats = {
                'run': i,
                'seed': current_seed,
                'n_agents': len(df)
            }
            
            # Add statistics for each decision output
            for col in df.columns:
                if col not in ['Assigned Allowance Level', 'Group_experiment', 'Honesty_Humility', 
                              'Study Program', 'TWT+Sospeso [=AW2+AX2]{Periods 1+2}']:
                    if pd.api.types.is_numeric_dtype(df[col]):
                        run_stats[f'{col}_mean'] = df[col].mean()
                        run_stats[f'{col}_std'] = df[col].std()
                        run_stats[f'{col}_min'] = df[col].min()
                        run_stats[f'{col}_max'] = df[col].max()
            
            mc_results.append(run_stats)
            individual_files.append(result_file)
            
            # Show key statistic (donation_default if available)
            if 'donation_default' in df.columns:
                print(f"donation_mean={df['donation_default'].mean():.4f}")
            else:
                print("completed")
                
        except subprocess.CalledProcessError as e:
            print(f"ERROR: {e}")
            continue
        except Exception as e:
            print(f"ERROR: {e}")
            continue
    
    if not mc_results:
        print("No successful runs completed!")
        return 1
    
    # Convert to DataFrame for analysis
    mc_df = pd.DataFrame(mc_results)
    
    print("\n" + "=" * 50)
    print(f"Monte-Carlo study completed: {len(mc_results)} successful runs")
    
    # Generate summary statistics
    summary_stats = {}
    
    # Focus on donation_default if available
    if 'donation_default_mean' in mc_df.columns:
        donation_means = mc_df['donation_default_mean']
        summary_stats['donation_default'] = {
            'mean': donation_means.mean(),
            'std': donation_means.std(),
            'p2.5': np.percentile(donation_means, 2.5),
            'p97.5': np.percentile(donation_means, 97.5),
            'runs': len(donation_means)
        }
        
        print(f"\nDonation Default Summary (across {len(donation_means)} runs):")
        print(f"  Mean: {summary_stats['donation_default']['mean']:.6f}")
        print(f"  Std:  {summary_stats['donation_default']['std']:.6f}")
        print(f"  95% CI: [{summary_stats['donation_default']['p2.5']:.6f}, {summary_stats['donation_default']['p97.5']:.6f}]")
    
    # Save detailed MC results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    decision_suffix = f"_{args.decision}" if args.decision else "_all"
    
    mc_detailed_path = output_dir / f"mc_detailed_runs{args.runs}_agents{args.agents}{decision_suffix}_{timestamp}.csv"
    mc_df.to_csv(mc_detailed_path, index=False)
    
    # Save summary statistics
    mc_summary_path = output_dir / f"mc_summary_runs{args.runs}_agents{args.agents}{decision_suffix}_{timestamp}.csv"
    
    summary_rows = []
    for decision, stats in summary_stats.items():
        summary_rows.append({
            'decision': decision,
            'mean': stats['mean'],
            'std': stats['std'],
            'p2.5': stats['p2.5'],
            'p97.5': stats['p97.5'],
            'runs': stats['runs'],
            'agents_per_run': args.agents
        })
    
    if summary_rows:
        pd.DataFrame(summary_rows).to_csv(mc_summary_path, index=False)
        print(f"\nSummary saved to: {mc_summary_path}")
    
    print(f"Detailed results saved to: {mc_detailed_path}")
    
    # Save configuration for reproducibility
    config_path = output_dir / f"mc_config_runs{args.runs}_agents{args.agents}{decision_suffix}_{timestamp}.json"
    config = {
        'agents_per_run': args.agents,
        'num_runs': args.runs,
        'base_seed': args.base_seed,
        'decision': args.decision,
        'seeds_used': list(range(args.base_seed, args.base_seed + args.runs)),
        'timestamp': timestamp
    }
    
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"Configuration saved to: {config_path}")
    
    # Clean up individual files if requested
    if not args.keep_individual and not args.summary_only:
        print(f"\nCleaning up {len(individual_files)} individual run files...")
        for file_path in individual_files:
            try:
                Path(file_path).unlink()
            except Exception as e:
                print(f"Warning: Could not delete {file_path}: {e}")
    
    print("\n✅ Monte-Carlo study completed successfully!")
    return 0

if __name__ == "__main__":
    exit(main())