# scripts/run_simulation.py
import argparse
import sys
from pathlib import Path
import pandas as pd
from datetime import datetime

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.orchestrator import Orchestrator

def main():
    parser = argparse.ArgumentParser(description='Run AI agent simulation')
    parser.add_argument('--agents', type=int, default=1000,
                       help='Number of synthetic agents to generate (default: 1000)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility (default: 42)')
    parser.add_argument('--decision', type=str, default=None,
                       help='Run only specific decision (default: run all)')
    parser.add_argument('--output-dir', type=str, default='outputs',
                       help='Output directory (default: outputs)')
    parser.add_argument('--format', choices=['parquet', 'csv'], default='parquet',
                       help='Output format (default: parquet)')
    
    args = parser.parse_args()
    
    # Initialize orchestrator
    print("Initializing simulation...")
    orchestrator = Orchestrator()
    
    if args.decision:
        available = orchestrator.get_available_decisions()
        if args.decision not in available:
            print(f"Error: Decision '{args.decision}' not available.")
            print(f"Available decisions: {', '.join(available)}")
            return 1
        print(f"Running single decision: {args.decision}")
    else:
        print("Running all 13 decisions")
    
    # Run simulation
    print(f"Generating {args.agents} synthetic agents with seed {args.seed}...")
    
    try:
        results_df = orchestrator.run_simulation(
            n_agents=args.agents,
            seed=args.seed,
            single_decision=args.decision
        )
        
        # Create output directory
        output_dir = Path(args.output_dir)
        output_dir.mkdir(exist_ok=True)
        
        # Generate output filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        decision_suffix = f"_{args.decision}" if args.decision else "_all"
        filename = f"simulation_seed{args.seed}_agents{args.agents}{decision_suffix}_{timestamp}"
        
        # Save results
        if args.format == 'parquet':
            output_path = output_dir / f"{filename}.parquet"
            results_df.to_parquet(output_path, index=False)
        else:
            output_path = output_dir / f"{filename}.csv"
            results_df.to_csv(output_path, index=False)
        
        print(f"\n✅ Simulation completed!")
        print(f"Results saved to: {output_path}")
        print(f"Shape: {results_df.shape}")
        
        # Show summary statistics for donation_default if it was computed
        if 'donation_default' in results_df.columns:
            donation_stats = results_df['donation_default'].describe()
            print(f"\nDonation Default Summary:")
            print(f"  Mean: {donation_stats['mean']:.4f}")
            print(f"  Std:  {donation_stats['std']:.4f}")
            print(f"  Min:  {donation_stats['min']:.4f}")
            print(f"  Max:  {donation_stats['max']:.4f}")
        
        # Show available columns
        print(f"\nOutput columns: {list(results_df.columns)}")
        
        return 0
        
    except Exception as e:
        print(f"Error during simulation: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit(main())