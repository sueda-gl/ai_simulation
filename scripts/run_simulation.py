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
    parser.add_argument('--decision', type=str, action='append', default=None,
                       help='Run specific decision(s). Can be specified multiple times. (default: run all)')
    parser.add_argument('--output-dir', type=str, default='outputs',
                       help='Output directory (default: outputs)')
    parser.add_argument('--format', choices=['parquet', 'csv'], default='parquet',
                       help='Output format (default: parquet)')
    parser.add_argument('--anchor-observed', type=float, default=0.75,
                       help='Weight on observed prosocial score for anchor (default: 0.75)')
    parser.add_argument('--population-mode', type=str, default='copula',
                       choices=['copula', 'documentation', 'baseline', 'depvar'],
                       help='Population generation mode (default: copula)')
    parser.add_argument('--income-mode', type=str, default='categorical',
                       choices=['categorical', 'continuous'],
                       help='Income specification mode (default: categorical)')
    
    args = parser.parse_args()
    
    # Initialize appropriate orchestrator based on population mode
    print("Initializing simulation...")
    print(f"Population mode: {args.population_mode}")
    print(f"Income specification: {args.income_mode}")
    
    if args.population_mode == 'documentation':
        from src.orchestrator_doc_mode import OrchestratorDocMode
        orchestrator = OrchestratorDocMode()
    elif args.population_mode == 'baseline':
        from src.orchestrator_baseline import OrchestratorBaseline
        orchestrator = OrchestratorBaseline()
    elif args.population_mode == 'depvar':
        from src.orchestrator_depvar import OrchestratorDepVar
        orchestrator = OrchestratorDepVar()
    else:  # copula (default)
        orchestrator = Orchestrator()
    
    # Apply anchor weights and income mode if donation_default is in scope
    if hasattr(orchestrator, 'config') and 'donation_default' in orchestrator.config:
        orchestrator.config['donation_default']['anchor_weights']['observed'] = args.anchor_observed
        orchestrator.config['donation_default']['anchor_weights']['predicted'] = 1 - args.anchor_observed
        print(f"Using anchor weights: {args.anchor_observed:.2f} observed | {1 - args.anchor_observed:.2f} predicted")
        
        # Set income mode
        if 'regression' not in orchestrator.config['donation_default']:
            orchestrator.config['donation_default']['regression'] = {}
        orchestrator.config['donation_default']['regression']['income_mode'] = args.income_mode
        
        if 'regression_coefficients' not in orchestrator.config['donation_default']:
            orchestrator.config['donation_default']['regression_coefficients'] = {}
        orchestrator.config['donation_default']['regression_coefficients']['income_mode'] = args.income_mode
        print(f"Income specification mode: {args.income_mode}")
    
    if args.decision:
        # Check if orchestrator has method to get available decisions
        if hasattr(orchestrator, 'get_available_decisions'):
            available = orchestrator.get_available_decisions()
            for decision in args.decision:
                if decision not in available:
                    print(f"Error: Decision '{decision}' not available.")
                    print(f"Available decisions: {', '.join(available)}")
                    return 1
        print(f"Running decisions: {', '.join(args.decision)}")
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
        # Create decision suffix for filename
        if args.decision is None:
            decision_suffix = "_all"
        elif len(args.decision) == 1:
            decision_suffix = f"_{args.decision[0]}"
        else:
            decision_suffix = f"_{len(args.decision)}decisions"
        filename = f"simulation_seed{args.seed}_agents{args.agents}{decision_suffix}_{timestamp}"
        
        # Save results
        if args.format == 'parquet':
            output_path = output_dir / f"{filename}.parquet"
            
            # Prepare DataFrame for parquet saving
            # Parquet can't handle complex nested structures, so convert purchase_requests to JSON
            df_to_save = results_df.copy()
            if 'purchase_requests' in df_to_save.columns:
                import json
                df_to_save['purchase_requests'] = df_to_save['purchase_requests'].apply(
                    lambda x: json.dumps(x) if isinstance(x, (list, dict)) else str(x)
                )
            
            df_to_save.to_parquet(output_path, index=False)
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