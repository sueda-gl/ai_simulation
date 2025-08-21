# Phase D: Final Regression – Factor Scores + Controls

**Dependent variable**: Prosocial_DV  
Observations: 280  
R²: 0.078 (Adj-R² 0.029)

## Predictors
Factor scores included:
- Factor1_F1_score
- Factor2_F1_score
- Factor3_F1_score
- Factor4_F1_score

Controls retained: Group dummies, TotalAllowance_std, Age_std, Education dummies, StudyProgram dummies, Foreign_binary, TopPlatform dummies (as available).

Expected prosociality (fitted values) saved to `expected_prosociality.csv`.

## Period sensitivity
``period_sensitivity_table.csv`` contains p-values for each predictor using the Period 1 and Period 2 components individually.  Any factor/control that is significant (p < 0.10) in at least one period is highlighted for discussion.

Ready for Phase E (mapping predicted prosociality to default donation rates).
