#!/usr/bin/env python3
"""
Phase C: SEM/MIMIC Approach - Proper inclusion of DV in factor modeling

This implements the theoretically correct approach where:
1. Survey items load on latent factors (measurement model)
2. DV is regressed on the latent factors + controls (structural model)
3. Factor weights are discovered in a way that reflects the DV relationship

This is the "correct way to include the DV" as requested by the professor.
"""

import pandas as pd
import numpy as np
from semopy import Model
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

def load_data():
    """Load the prepared analysis table"""
    return pd.read_csv("analysis_table.csv")

def define_measurement_model():
    """
    Define the measurement model: survey items → latent factors
    Based on our theory-driven bundles from previous analysis
    """
    
    # Define conceptual bundles (same as before, but now in SEM syntax)
    model_syntax = """
    # Measurement Model (Survey Items -> Latent Factors)
    
    # Prosocial Orientation Factor
    PROSOCIAL =~ SVO_type_std + Prosocial_Motivation_std + Dictator_1_std
    
    # Honesty/Integrity Factor  
    HONESTY =~ Agreeable_std + Integrity_Honesty_std + Kindness_std + Honesty_Humility_std
    
    # Big Five Personality Factor
    BIGFIVE =~ NeuroticismBig5_std + ExtraversionBig5_std + OpennessBig5_std + ConscientiousnessBig5_std
    
    # Affect/Well-being Factor
    AFFECT =~ PosAffect_std + LifeSatis_std + SubHappy_std + HumanVS_std
    
    # Political/Social Ideology Factor
    IDEOLOGY =~ ClassSystemJust_std + EconomicBelief_std + SupportEquality_std + SocialOrientation_std + Egalitarianism_std + RSDO_std + Religious_std + ReligiousService_std + IndividualIncome_std
    
    # Structural Model (Latent Factors + Controls -> DV)
    Prosocial_DV ~ PROSOCIAL + HONESTY + BIGFIVE + AFFECT + IDEOLOGY + TotalAllowance_std + Age_std + Foreign_binary + Group_MidSub + Group_NoSub + Education_2 + StudyProgram_Incoming + StudyProgram_Law_5_year_Program + StudyProgram_UG_3_year_Program + TopPlatform_1
    """
    
    return model_syntax

def run_sem_analysis(df):
    """
    Run the full SEM/MIMIC model
    """
    
    print("="*60)
    print("PHASE C: SEM/MIMIC APPROACH")
    print("="*60)
    
    # Prepare data - select only variables needed for SEM
    survey_vars = [
        'SVO_type_std', 'Prosocial_Motivation_std', 'Dictator_1_std',
        'Agreeable_std', 'Integrity_Honesty_std', 'Kindness_std', 'Honesty_Humility_std',
        'NeuroticismBig5_std', 'ExtraversionBig5_std', 'OpennessBig5_std', 'ConscientiousnessBig5_std',
        'PosAffect_std', 'LifeSatis_std', 'SubHappy_std', 'HumanVS_std',
        'ClassSystemJust_std', 'EconomicBelief_std', 'SupportEquality_std', 
        'SocialOrientation_std', 'Egalitarianism_std', 'RSDO_std', 
        'Religious_std', 'ReligiousService_std', 'IndividualIncome_std'
    ]
    
    control_vars = [
        'TotalAllowance_std', 'Age_std', 'Foreign_binary', 
        'Group_MidSub', 'Group_NoSub', 'Education_2',
        'StudyProgram_Incoming', 'StudyProgram_Law_5_year_Program', 
        'StudyProgram_UG_3_year_Program', 'TopPlatform_1'
    ]
    
    dv = ['Prosocial_DV']
    
    all_vars = survey_vars + control_vars + dv
    
    # Check which variables exist in the data
    available_vars = [var for var in all_vars if var in df.columns]
    missing_vars = [var for var in all_vars if var not in df.columns]
    
    print(f"Variables available: {len(available_vars)}/{len(all_vars)}")
    if missing_vars:
        print(f"Missing variables: {missing_vars}")
    
    # Use available variables for analysis
    sem_data = df[available_vars].dropna()
    print(f"Complete cases for SEM: {len(sem_data)}")
    
    # Adjust model syntax based on available variables
    adjusted_syntax = adjust_model_syntax(available_vars)
    
    print("\nModel Specification:")
    print(adjusted_syntax)
    
    # Fit the SEM model
    print("\nFitting SEM model...")
    try:
        model = Model(adjusted_syntax)
        model.fit(sem_data)
        
        # Print model fit statistics
        print("\n" + "="*40)
        print("MODEL FIT STATISTICS")
        print("="*40)
        
        fit_stats = model.inspect()
        print(f"Chi-square: {fit_stats.get('chi2', 'N/A')}")
        print(f"Degrees of freedom: {fit_stats.get('dof', 'N/A')}")
        print(f"P-value: {fit_stats.get('pvalue', 'N/A')}")
        print(f"CFI: {fit_stats.get('cfi', 'N/A')}")
        print(f"RMSEA: {fit_stats.get('rmsea', 'N/A')}")
        print(f"SRMR: {fit_stats.get('srmr', 'N/A')}")
        
        # Print parameter estimates
        print("\n" + "="*40)
        print("PARAMETER ESTIMATES")
        print("="*40)
        
        params = model.inspect(what='est')
        params_df = pd.DataFrame(params).round(4)
        print(params_df.to_string(index=False))
        
        # Save detailed results
        params_df.to_csv("sem_parameter_estimates.csv", index=False)
        
        # Extract factor scores if possible
        try:
            factor_scores = model.predict(sem_data)
            factor_scores_df = pd.DataFrame(factor_scores, index=sem_data.index)
            factor_scores_df.to_csv("sem_factor_scores.csv")
            print(f"\nFactor scores saved to: sem_factor_scores.csv")
        except Exception as e:
            print(f"Could not extract factor scores: {e}")
        
        # Generate comprehensive report
        generate_sem_report(model, fit_stats, params_df, len(sem_data))
        
        return model, fit_stats, params_df
        
    except Exception as e:
        print(f"SEM fitting failed: {e}")
        print("This might be due to model complexity or identification issues.")
        print("Consider simplifying the measurement model.")
        return None, None, None

def adjust_model_syntax(available_vars):
    """
    Adjust the model syntax based on available variables
    """
    
    # Check which bundle variables are available
    prosocial_vars = [v for v in ['SVO_type_std', 'Prosocial_Motivation_std', 'Dictator_1_std'] if v in available_vars]
    honesty_vars = [v for v in ['Agreeable_std', 'Integrity_Honesty_std', 'Kindness_std', 'Honesty_Humility_std'] if v in available_vars]
    bigfive_vars = [v for v in ['NeuroticismBig5_std', 'ExtraversionBig5_std', 'OpennessBig5_std', 'ConscientiousnessBig5_std'] if v in available_vars]
    affect_vars = [v for v in ['PosAffect_std', 'LifeSatis_std', 'SubHappy_std', 'HumanVS_std'] if v in available_vars]
    ideology_vars = [v for v in ['ClassSystemJust_std', 'EconomicBelief_std', 'SupportEquality_std', 'SocialOrientation_std', 'Egalitarianism_std', 'RSDO_std', 'Religious_std', 'ReligiousService_std', 'IndividualIncome_std'] if v in available_vars]
    
    control_vars = [v for v in ['TotalAllowance_std', 'Age_std', 'Foreign_binary', 'Group_MidSub', 'Group_NoSub', 'Education_2', 'StudyProgram_Incoming', 'StudyProgram_Law_5_year_Program', 'StudyProgram_UG_3_year_Program', 'TopPlatform_1'] if v in available_vars]
    
    # Build measurement model
    measurement_lines = []
    structural_predictors = []
    
    if len(prosocial_vars) >= 2:
        measurement_lines.append(f"PROSOCIAL =~ {' + '.join(prosocial_vars)}")
        structural_predictors.append("PROSOCIAL")
    
    if len(honesty_vars) >= 2:
        measurement_lines.append(f"HONESTY =~ {' + '.join(honesty_vars)}")
        structural_predictors.append("HONESTY")
    
    if len(bigfive_vars) >= 2:
        measurement_lines.append(f"BIGFIVE =~ {' + '.join(bigfive_vars)}")
        structural_predictors.append("BIGFIVE")
    
    if len(affect_vars) >= 2:
        measurement_lines.append(f"AFFECT =~ {' + '.join(affect_vars)}")
        structural_predictors.append("AFFECT")
    
    if len(ideology_vars) >= 2:
        measurement_lines.append(f"IDEOLOGY =~ {' + '.join(ideology_vars)}")
        structural_predictors.append("IDEOLOGY")
    
    # Build structural model
    all_predictors = structural_predictors + control_vars
    if all_predictors and 'Prosocial_DV' in available_vars:
        structural_line = f"Prosocial_DV ~ {' + '.join(all_predictors)}"
    else:
        structural_line = "# No valid structural model could be built"
    
    # Combine into full model
    full_model = "\n".join(["# Measurement Model"] + measurement_lines + ["", "# Structural Model", structural_line])
    
    return full_model

def generate_sem_report(model, fit_stats, params_df, n_obs):
    """
    Generate a comprehensive markdown report
    """
    
    report_lines = [
        "# Phase C: SEM/MIMIC Analysis Report",
        "",
        "## Model Approach",
        "This analysis uses **Structural Equation Modeling (SEM)** with a MIMIC approach:",
        "- **Measurement Model**: Survey items load on latent factors",
        "- **Structural Model**: DV regressed on latent factors + controls",
        "- **Advantage**: Factor weights discovered considering DV relationship",
        "",
        f"**Sample Size**: {n_obs} complete cases",
        "",
        "## Model Fit Statistics",
        f"- **Chi-square**: {fit_stats.get('chi2', 'N/A')}",
        f"- **Degrees of Freedom**: {fit_stats.get('dof', 'N/A')}",
        f"- **P-value**: {fit_stats.get('pvalue', 'N/A')}",
        f"- **CFI**: {fit_stats.get('cfi', 'N/A')} (> 0.95 = excellent fit)",
        f"- **RMSEA**: {fit_stats.get('rmsea', 'N/A')} (< 0.06 = excellent fit)",
        f"- **SRMR**: {fit_stats.get('srmr', 'N/A')} (< 0.08 = good fit)",
        "",
        "## Parameter Estimates",
        "See `sem_parameter_estimates.csv` for detailed results.",
        "",
        "### Factor Loadings (Measurement Model)",
        "These show how strongly each survey item loads on its latent factor.",
        "",
        "### Structural Coefficients",
        "These show how latent factors predict the prosocial DV.",
        "",
        "## Advantages of SEM/MIMIC Approach",
        "1. **Theoretically Correct**: DV influences factor weight discovery",
        "2. **Measurement Error**: Accounts for unreliability in survey measures",
        "3. **Simultaneous Estimation**: Measurement and structural parameters estimated together",
        "4. **Fit Assessment**: Provides model fit indices for validation",
        "5. **Standard Errors**: Properly propagates uncertainty from measurement to structural model",
        "",
        "## Next Steps",
        "- Use the structural coefficients for behavioral prediction",
        "- Extract factor scores for individual-level analysis",
        "- Compare with previous EFA approaches",
    ]
    
    with open("phase_c_sem_mimic.md", "w") as f:
        f.write("\n".join(report_lines))
    
    print(f"\nReport saved to: phase_c_sem_mimic.md")

def main():
    """Main execution function"""
    
    # Load data
    print("Loading analysis table...")
    df = load_data()
    
    # Run SEM analysis
    model, fit_stats, params = run_sem_analysis(df)
    
    if model is not None:
        print("\n✅ SEM/MIMIC analysis completed successfully!")
        print("📊 Check sem_parameter_estimates.csv for detailed results")
        print("📝 Check phase_c_sem_mimic.md for full report")
    else:
        print("\n❌ SEM analysis failed - see error messages above")

if __name__ == "__main__":
    main() 