# Phase A: Data Integration & Preparation

## Summary
- **Field experiment data**: 280 participants, 182 variables
- **Survey data**: Data merged successfully by Participant ID
- **Final analysis dataset**: 280 complete cases

## Data Integration
- Merge type: Inner join on Participant ID
- Participants in both datasets: 280
- No duplicate participant IDs found

## Dependent Variable (Prosocial_DV)
- Source: TWT+Sospeso [=AW2+AX2] across Periods 1+2
- Mean: 3.36
- Standard deviation: 9.90
- Range: 0.0 to 112.0
- Zero values: 176 (62.9%)

## Control Variables Encoded
- **Group**: Categorical with dummy variables (baseline: Fixed)
- **Total Allowance**: Standardized continuous variable
- **Age**: Standardized continuous variable
- **Education**: Categorical dummy variables
- **Study Program Category**: Categorical dummy variables
- **Foreign**: Binary variable
- **TopPlatform**: Categorical dummy variables

## Survey Predictors Standardized
Total standardized predictors: 24

- SVO_type
- Prosocial_Motivation
- Agreeable
- Integrity_Honesty
- Kindness
- Honesty_Humility
- NeuroticismBig5
- ExtraversionBig5
- OpennessBig5
- ConscientiousnessBig5
- HumanVS
- PosAffect
- LifeSatis
- SubHappy
- ClassSystemJust
- EconomicBelief
- SupportEquality
- SocialOrientation
- Egalitarianism
- RSDO
- Religious
- ReligiousService
- IndividualIncome
- Dictator_1

## Missing Data
- Complete case analysis used
- Rows lost to missing data: 0

## Files Generated
- `analysis_table.csv`: Clean analysis dataset ready for modeling
- `dv_distribution.png`: Histogram of dependent variable
- `phase_a_data_preparation.md`: This report

## Next Steps
Ready for Phase B: Screening regression to identify significant predictors.
