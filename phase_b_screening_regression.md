# Phase B: Screening Regression

**Model**: OLS with HC3 robust s.e.  
Observations: 280  
R²: 0.117 (Adj. R² -0.002)  
F-statistic p-value: 0.97

## Kept survey predictors (p < 0.10)
- TotalAllowance
- SVO_type

## Dropped survey predictors
- Agreeable
- Religious
- HumanVS
- ExtraversionBig5
- LifeSatis
- SupportEquality
- SocialOrientation
- SubHappy
- RSDO
- Age
- … and 14 others

All controls (Group, TotalAllowance, Age, Education, StudyProgram, Foreign, TopPlatform) are retained regardless of significance, per the professor’s instructions.

Files generated:
- `screening_regression_table.csv`
- `significant_predictors.png`

Ready to proceed to Phase C (Exploratory Factor Analysis) on the kept predictors plus the DV.
