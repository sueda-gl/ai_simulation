#!/usr/bin/env python3
"""
Script to identify and fix copula calculation differences.
This checks for common configuration issues between environments.
"""

import yaml
from pathlib import Path
import subprocess
import sys

def check_and_fix_issues():
    """Check for common issues and suggest fixes."""
    
    print("=" * 80)
    print("CHECKING FOR COPULA CALCULATION ISSUES")
    print("=" * 80)
    
    issues_found = []
    
    # 1. Check git status for uncommitted changes
    print("\n1. Checking for uncommitted changes...")
    try:
        result = subprocess.run(['git', 'status', '--porcelain'], 
                              capture_output=True, text=True)
        if result.stdout:
            print("⚠️  Uncommitted changes found:")
            print(result.stdout)
            issues_found.append("uncommitted_changes")
        else:
            print("✅ No uncommitted changes")
    except:
        print("❌ Could not check git status")
    
    # 2. Check configuration file
    print("\n2. Checking configuration...")
    config_path = Path("config/decisions.yaml")
    if config_path.exists():
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        donation_config = config.get('donation_default', {})
        
        # Check in_copula flag
        in_copula = donation_config.get('stochastic', {}).get('in_copula', False)
        print(f"   - stochastic.in_copula: {in_copula}")
        if not in_copula:
            print("     ⚠️  Copula mode is NOT using stochastic component")
            print("     This results in lower donation rates")
            issues_found.append("in_copula_false")
        
        # Check income_mode
        regression = donation_config.get('regression', {})
        if 'income_mode' not in regression:
            print("   - income_mode: NOT SET (will default to 'categorical')")
            print("     ⚠️  Income mode not explicitly set in config")
            issues_found.append("income_mode_not_set")
        else:
            print(f"   - income_mode: {regression['income_mode']}")
    
    # 3. Suggest fixes
    print("\n" + "=" * 80)
    print("SUGGESTED FIXES:")
    print("=" * 80)
    
    if "uncommitted_changes" in issues_found:
        print("\n1. COMMIT OR SYNC YOUR CHANGES:")
        print("   Your local version has uncommitted changes that may not be deployed.")
        print("   Run: git add -A && git commit -m 'Update copula calculations'")
        print("   Then: git push origin main")
    
    if "in_copula_false" in issues_found:
        print("\n2. ENABLE STOCHASTIC COMPONENT FOR COPULA (if desired):")
        print("   Edit config/decisions.yaml, line 48:")
        print("   Change: in_copula: false")
        print("   To:     in_copula: true")
        print("   This will make copula mode use the same stochastic draw as documentation mode.")
    
    if "income_mode_not_set" in issues_found:
        print("\n3. SET EXPLICIT INCOME MODE:")
        print("   Add to config/decisions.yaml under donation_default.regression:")
        print("   income_mode: categorical  # or 'continuous'")
    
    print("\n4. ENSURE CONSISTENCY:")
    print("   - Make sure deployed version has latest trait_model.pkl")
    print("   - Verify all dependencies match between environments") 
    print("   - Consider setting fixed seeds in Streamlit for reproducibility")
    
    print("\n5. DEBUGGING IN STREAMLIT:")
    print("   - Check the 'Add Normal(anchor, σ) draw to Copula runs' checkbox")
    print("   - This dynamically enables stochastic component")
    print("   - Compare results with and without this option")

if __name__ == "__main__":
    check_and_fix_issues()
