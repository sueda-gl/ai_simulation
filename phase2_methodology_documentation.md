# Phase 2: Individual Donation Rate Estimation - Methodology Documentation

## Executive Summary

This document outlines the methodology for estimating personalized donation rates based on experimental prosocial behavior and demographic characteristics. We developed a two-component scoring system that blends observed behavior with model-based predictions to generate deployment-ready donation probabilities.

## 1. Research Objective

**Primary Goal**: Create individual-level donation rate predictions (0-100%) that can be deployed in real-world applications without requiring experimental data from new participants.

**Key Innovation**: Instead of relying on external coefficients from other studies, we calibrate our model using our own experimental data and blend observed behavior with demographic expectations.

## 2. Data Sources

- **Survey Data**: `Student Survey Results - Period 1.xlsx` (n=280)
  - Honesty-Humility scores from HEXACO personality inventory
  - Individual income levels (€)
  - Study program categories
  
- **Experimental Data**: `Student Experiment Results - Period 1-2.xlsx` (n=280)
  - Prosocial behavior: TWT+Sospeso donations across both experimental periods
  - Study program categories (for validation/coalescing)

## 3. Model Specification

### 3.1 Dependent Variable
```
Y_i = TWT+Sospeso [=AW2+AX2]{Periods 1+2}
```
Where Y_i represents the total prosocial donations (tokens + sospeso) for participant i across both experimental periods. This variable:
- Range: [0, 112] in our dataset
- Mean: 3.36, Median: 0.00, SD: 9.90
- **Note**: This is pre-existing data from the experiment, not calculated by regression

### 3.2 Predictor Variables

**Continuous Predictor:**
```
z_hh_i = (HH_i - μ_HH) / σ_HH
```
Where:
- HH_i = Raw Honesty-Humility score for participant i
- μ_HH = Sample mean of HH scores (3.392)
- σ_HH = Sample standard deviation of HH scores (0.559)

**Categorical Predictors:**
1. **Income Quintiles** (based on Eurostat income bands):
   ```
   Income_Cat_i ∈ {1, 2, 3, 4, 5}
   Bin edges: [-∞, 12617, 18030, 23567, 31487, +∞] €
   ```

2. **Study Program Categories**:
   ```
   Study_Prog_i ∈ {G 2-year Program, UG 3-year Program, Law 5-year Program, Incoming}
   ```

### 3.3 Regression Model
```
Y_i = β_0 + β_1 × z_hh_i + Σ(j=2 to 5) γ_j × I(Income_Cat_i = j) + Σ(k=1 to 3) δ_k × I(Study_Prog_i = k) + ε_i
```

Where:
- β_0 = Intercept (baseline: Income Q1, G 2-year Program)
- β_1 = Coefficient for standardized Honesty-Humility
- γ_j = Coefficients for income quintiles 2-5 (Q1 as reference)
- δ_k = Coefficients for study programs (G 2-year as reference)
- ε_i = Error term
- I(·) = Indicator function

## 4. Estimation Results

### 4.1 Model Coefficients
```
Ŷ_i = 4.57 + 0.50 × z_hh_i + 0.00 × I(Income_Q2-Q5) - 1.87 × I(UG_3yr) - 2.30 × I(Law_5yr) - 5.23 × I(Incoming)
```

| Parameter | Estimate | Std Error | t-value | p-value | 95% CI |
|-----------|----------|-----------|---------|---------|---------|
| β_0 (Intercept) | 4.57 | 0.98 | 4.66 | <0.001*** | [2.64, 6.50] |
| β_1 (z_hh) | 0.50 | 0.60 | 0.84 | 0.401 | [-0.67, 1.68] |
| δ_1 (UG 3-year) | -1.87 | 1.25 | -1.50 | 0.134 | [-4.32, 0.58] |
| δ_2 (Law 5-year) | -2.30 | 3.45 | -0.67 | 0.506 | [-9.10, 4.50] |
| δ_3 (Incoming) | -5.23 | 9.99 | -0.52 | 0.601 | [-24.90, 14.44] |

### 4.2 Model Diagnostics
- **R-squared**: 0.012 (1.2% of variance explained)
- **Adjusted R-squared**: -0.003
- **F-statistic**: 0.821 (p = 0.512)
- **RMSE**: 9.87
- **Sample size**: 280 observations

**Interpretation**: The low R² indicates that donation behavior is highly individual and not well-predicted by demographics alone, justifying our approach of blending observed and predicted behavior.

## 5. Two-Component Scoring System

### 5.1 Component 1: Observed Behavior (Rescaled)
```
Obs_0100_i = 100 × (Y_i - min(Y)) / (max(Y) - min(Y))
```
Where:
- min(Y) = 0, max(Y) = 112
- Result: Observed donation rate on 0-100 scale

### 5.2 Component 2: Model Prediction (Rescaled)
```
Other_Score_i = β̂_0 + β̂_1 × z_hh_i + Σγ̂_j × I(Income_Cat_i = j) + Σδ̂_k × I(Study_Prog_i = k)

Other_0100_i = 100 × (Other_Score_i - min(Other_Score)) / (max(Other_Score) - min(Other_Score))
```

### 5.3 Blended Anchors
For policy weight w ∈ {0.25, 0.50, 0.75}:
```
Anchor_w_i = w × Obs_0100_i + (1-w) × Other_0100_i
```

**Interpretation**:
- w = 0.25: 25% weight on observed behavior, 75% on demographic model
- w = 0.50: Equal weighting (default)  
- w = 0.75: 75% weight on observed behavior, 25% on demographic model

### 5.4 Stochastic Predictions
```
Pred_w_i = max(0, min(100, Anchor_w_i + ε_noise_i))

where ε_noise_i ~ N(0, σ_noise²)
```

**Noise specification**:
```
σ_noise = NOISE_SCALE × σ_0100
σ_0100 = 100 × RMSE / (max(Y) - min(Y)) = 100 × 9.87 / 112 = 8.81
σ_noise = 0.5 × 8.81 = 4.41
```

## 6. Results Summary

### 6.1 Component Statistics (n=280)
| Component | Mean | Std Dev | Min | Median | Max |
|-----------|------|---------|-----|--------|-----|
| Observed Behavior (0-100) | 3.00 | 8.84 | 0.00 | 0.00 | 100.00 |
| Model Prediction (0-100) | 57.41 | 18.39 | 0.00 | 52.67 | 100.00 |
| Anchor 50% (Blend) | 30.20 | 10.63 | 0.00 | 27.04 | 86.55 |
| Final Prediction 50% | 29.85 | 11.54 | 0.00 | 28.00 | 88.95 |

### 6.2 Prediction Comparison by Weight
| Weight | Mean | Std Dev | Interpretation |
|--------|------|---------|----------------|
| pred_25 | 43.9% | 15.0 | Model-focused (conservative on observed behavior) |
| pred_50 | 29.9% | 11.5 | Balanced blend (default deployment) |
| pred_75 | 16.8% | 9.7 | Behavior-focused (realistic based on observations) |

### 6.3 Study Program Effects
| Program | n | Observed Mean | Predicted Mean |
|---------|---|---------------|----------------|
| G 2-year Program | 102 | 4.59 | 39.99% |
| UG 3-year Program | 168 | 2.70 | 23.67% |
| Law 5-year Program | 9 | 2.11 | 15.49% |
| Incoming | 1 | 0.00 | 0.00% |

## 7. Methodological Justifications

### 7.1 Why Blend Observed and Predicted?
1. **Merit Preservation**: Participants who actually donated should score higher
2. **Stability**: Pure observed behavior is noisy; demographic model provides baseline expectations
3. **Generalizability**: Model component helps predict behavior for new participants with similar characteristics
4. **Policy Flexibility**: Different weights allow testing various reward structures

### 7.2 Why These Predictors?
- **Honesty-Humility**: Established predictor of prosocial behavior in psychology literature
- **Income**: Economic capacity affects donation ability
- **Study Program**: Proxy for values, social environment, and future career paths
- **Note**: Experimental group deliberately excluded as it's not a stable personal characteristic

### 7.3 Why Add Stochastic Noise?
1. Prevents identical scores for participants with same demographics
2. Reflects natural variation in behavior
3. Enables probabilistic deployment in applications
4. Conservative noise level (0.5 × RMSE) maintains rank ordering while adding realism

## 8. Deployment Specifications

### 8.1 Required Inputs for New Participants
- Honesty-Humility score (HEXACO scale)
- Annual income (€) → converted to quintile
- Study program/academic background

### 8.2 Output
- Personalized donation probability (0-100%)
- Default recommendation: use `pred_50` (50% weight)
- Alternative weightings available based on policy preferences

### 8.3 Model Parameters (Saved for Deployment)
All coefficients, scaling parameters, bin edges, and noise specifications saved in `phase2_params.json` for consistent implementation.

## 9. Limitations and Future Directions

### 9.1 Current Limitations
- Low model R² suggests important predictors may be missing
- Sample consists entirely of students (income quintile 1)
- Cross-sectional design limits causal interpretation
- Experimental context may not generalize to real-world donations

### 9.2 Potential Improvements
- Include additional personality dimensions (Big Five)
- Test different functional forms (non-linear relationships)
- Validate on external samples
- Incorporate behavioral economics factors (framing, social norms)

## 10. Reproducibility

All analyses conducted in Python using:
- `pandas` for data manipulation
- `statsmodels` for regression analysis  
- `numpy` for numerical operations
- Random seed set to 42 for reproducible stochastic components

**Files generated**:
- `phase2_outputs.xlsx`: Individual-level predictions
- `phase2_params.json`: All model parameters for deployment
- `phase2_model_summary.txt`: Detailed regression output
- `phase2_pred_summary.xlsx`: Summary statistics by prediction type

---

*Generated: August 6, 2025*  
*Authors: Sueda & Andrei*