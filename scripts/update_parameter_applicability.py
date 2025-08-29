#!/usr/bin/env python3
"""
Script to easily update parameter applicability configuration.

Usage:
    python scripts/update_parameter_applicability.py --decision donation_default --add income_distribution --remove market_price
    python scripts/update_parameter_applicability.py --decision bid_value --list
    python scripts/update_parameter_applicability.py --all-decisions
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.parameter_applicability import param_manager

def list_decision_parameters(decision: str):
    """List current parameters for a decision"""
    summary = param_manager.get_decision_summary(decision)
    
    print(f"\n📊 Parameter Summary for '{decision}':")
    print(f"   Total Parameters: {summary['total_parameters']}")
    print(f"   Applicable: {summary['applicable_count']}")
    print(f"   Not Applicable: {summary['not_applicable_count']}")
    print(f"   Efficiency: {summary['applicability_ratio']:.0%}")
    
    if summary['reason']:
        print(f"   Reason: {summary['reason']}")
    
    print(f"\n✅ Applicable Parameters ({summary['applicable_count']}):")
    for param in summary['applicable_parameters']:
        print(f"   • {param}")
    
    print(f"\n❌ Not Applicable Parameters ({summary['not_applicable_count']}):")
    for param in summary['not_applicable_parameters']:
        print(f"   • {param}")

def list_all_decisions():
    """List all decisions and their parameter counts"""
    all_summaries = param_manager.get_all_decisions_summary()
    
    print("\n📊 All Decisions Parameter Summary:")
    print("-" * 80)
    
    for decision, summary in all_summaries.items():
        efficiency = f"{summary['applicability_ratio']:.0%}"
        print(f"{decision:<25} | Applicable: {summary['applicable_count']:>2} | "
              f"Not Applicable: {summary['not_applicable_count']:>2} | Efficiency: {efficiency:>4}")
    
    print("-" * 80)

def update_decision_parameters(decision: str, add_params: list, remove_params: list, reason: str = None):
    """Update parameters for a decision"""
    current_applicable = param_manager.get_applicable_parameters(decision)
    
    # Add new parameters
    updated_applicable = set(current_applicable)
    for param in add_params:
        if param in param_manager.all_parameters:
            updated_applicable.add(param)
            print(f"✅ Added '{param}' to {decision}")
        else:
            print(f"❌ Warning: '{param}' is not a recognized parameter")
    
    # Remove parameters
    for param in remove_params:
        if param in updated_applicable:
            updated_applicable.remove(param)
            print(f"🗑️ Removed '{param}' from {decision}")
        else:
            print(f"❌ Warning: '{param}' was not applicable for {decision}")
    
    # Get current reason if not provided
    if reason is None:
        current_config = param_manager.config.get('decisions', {}).get(decision, {})
        reason = current_config.get('not_applicable_reason', '')
    
    # Update the configuration
    param_manager.update_decision_applicability(
        decision=decision,
        applicable_parameters=list(updated_applicable),
        not_applicable_reason=reason
    )
    
    print(f"\n✅ Updated parameter applicability for '{decision}'")
    print(f"   New applicable count: {len(updated_applicable)}")

def main():
    parser = argparse.ArgumentParser(description="Update parameter applicability configuration")
    parser.add_argument('--decision', type=str, help='Decision to update')
    parser.add_argument('--add', nargs='*', default=[], help='Parameters to add as applicable')
    parser.add_argument('--remove', nargs='*', default=[], help='Parameters to remove from applicable')
    parser.add_argument('--reason', type=str, help='Reason for non-applicable parameters')
    parser.add_argument('--list', action='store_true', help='List current parameters for the decision')
    parser.add_argument('--all-decisions', action='store_true', help='List all decisions and their parameter counts')
    parser.add_argument('--available-params', action='store_true', help='List all available parameters')
    
    args = parser.parse_args()
    
    if args.all_decisions:
        list_all_decisions()
        return
    
    if args.available_params:
        print("\n📋 All Available Parameters:")
        categories = param_manager.parameter_categories
        for category, params in categories.items():
            print(f"\n{category.replace('_', ' ').title()}:")
            for param in params:
                print(f"   • {param}")
        return
    
    if not args.decision:
        print("❌ Error: --decision is required unless using --all-decisions or --available-params")
        parser.print_help()
        return
    
    if args.decision not in param_manager.config.get('decisions', {}):
        print(f"❌ Error: Decision '{args.decision}' not found in configuration")
        print("Available decisions:")
        for decision in param_manager.config.get('decisions', {}).keys():
            print(f"   • {decision}")
        return
    
    if args.list:
        list_decision_parameters(args.decision)
        return
    
    if args.add or args.remove:
        update_decision_parameters(args.decision, args.add, args.remove, args.reason)
        # Show updated parameters
        print("\n" + "="*50)
        list_decision_parameters(args.decision)
    else:
        print("❌ Error: Specify --add, --remove, or --list")
        parser.print_help()

if __name__ == "__main__":
    main()
