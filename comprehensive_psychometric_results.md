# Comprehensive Psychometric Results Report

**Prepared for Professor Review**  
*Complete Factor Analysis, Correlation, and Reliability Statistics*

---

## Executive Summary

This report provides comprehensive psychometric evidence across the two main methodological approaches employed in our prosocial behavior prediction study:

1. **Theory-Driven Bundle EFA** (Primary approach)
2. **Data-Driven EFA** (Alternative approach)

All analyses include factor loadings, correlation matrices, reliability coefficients (Cronbach's α), and model adequacy statistics as requested. Note that the dependent variable (Prosocial_DV) appears in all correlation matrices and factor loadings because it was explicitly included in each EFA bundle per the professor's instruction to "include the DV" in factor analysis.

---

## 1. THEORY-DRIVEN BUNDLE EFA RESULTS

### 1.1 PROSOCIAL Bundle

**Variables:** Prosocial_DV, SVO_type_std, Prosocial_Motivation_std, Dictator_1_std

**Correlation Matrix:**
```
                          Prosocial_DV  SVO_type_std  Prosocial_Motivation_std  Dictator_1_std
Prosocial_DV                     1.000         0.159                     0.058           0.052
SVO_type_std                     0.159         1.000                     0.265           0.319
Prosocial_Motivation_std         0.058         0.265                     1.000           0.279
Dictator_1_std                   0.052         0.319                     1.000           1.000
```

**Factor Loadings:**
```
Variable                     PROSOCIAL_F1    Communality (h²)
Prosocial_DV                     -0.172            0.030
SVO_type_std                     -0.595            0.354
Prosocial_Motivation_std         -0.469            0.220
Dictator_1_std                   -0.546            0.298
```

**Psychometric Statistics:**
- **Cronbach's α:** 0.548 (Poor reliability)
- **KMO Sampling Adequacy:** 0.622 (Adequate)
- **Bartlett's Test:** χ² = 69.855, p < 0.001 (Significant)
- **Eigenvalue:** 0.902
- **Variance Explained:** 22.5%

### 1.2 HONESTY Bundle

**Variables:** Prosocial_DV, Honesty_Humility_std, Integrity_Honesty_std, Kindness_std, Agreeable_std

**Correlation Matrix:**
```
                       Prosocial_DV  Honesty_Humility_std  Integrity_Honesty_std  Kindness_std  Agreeable_std
Prosocial_DV                  1.000                 0.052                  0.009         0.076          0.112
Honesty_Humility_std          0.052                 1.000                  0.375         0.360          0.617
Integrity_Honesty_std         0.009                 0.375                  1.000         0.390          0.424
Kindness_std                  0.076                 0.360                  0.390         1.000          0.526
Agreeable_std                 0.112                 0.617                  0.424         0.526          1.000
```

**Factor Loadings:**
```
Variable                     HONESTY_F1      Communality (h²)
Prosocial_DV                     0.098            0.010
Honesty_Humility_std             0.675            0.456
Integrity_Honesty_std            0.542            0.293
Kindness_std                     0.608            0.370
Agreeable_std                    0.866            0.750
```

**Psychometric Statistics:**
- **Cronbach's α:** 0.765 (Acceptable reliability)
- **KMO Sampling Adequacy:** 0.724 (Good)
- **Bartlett's Test:** χ² = 301.725, p < 0.001 (Significant)
- **Eigenvalue:** 1.879
- **Variance Explained:** 37.6%

### 1.3 BIGFIVE Bundle

**Variables:** Prosocial_DV, ExtraversionBig5_std, OpennessBig5_std, ConscientiousnessBig5_std, NeuroticismBig5_std

**Factor Loadings (Two Factors Extracted):**
```
Variable                     BIGFIVE_F1    BIGFIVE_F2    Communality (h²)
Prosocial_DV                     0.053         0.016            0.003
ExtraversionBig5_std             0.241         0.410            0.226
OpennessBig5_std                 0.965        -0.001            0.931
ConscientiousnessBig5_std        0.012         0.714            0.510
NeuroticismBig5_std              0.160        -0.483            0.259
```

**Psychometric Statistics:**
- **Cronbach's α:** 0.204 (Poor reliability)
- **KMO Sampling Adequacy:** < 0.6 (Bundle skipped in final analysis)
- **Eigenvalues:** F1 = 1.018, F2 = 0.912
- **Variance Explained:** F1 = 20.4%, F2 = 18.2%

### 1.4 AFFECT Bundle

**Variables:** Prosocial_DV, PosAffect_std, LifeSatis_std, SubHappy_std

**Correlation Matrix:**
```
               Prosocial_DV  PosAffect_std  LifeSatis_std  SubHappy_std
Prosocial_DV          1.000          0.015          0.062         0.091
PosAffect_std         0.015          1.000          0.399         0.496
LifeSatis_std         0.062          0.399          1.000         0.551
SubHappy_std          0.091          0.496          0.551         1.000
```

**Factor Loadings:**
```
Variable                     AFFECT_F1       Communality (h²)
Prosocial_DV                    -0.085            0.007
PosAffect_std                   -0.595            0.354
LifeSatis_std                   -0.665            0.442
SubHappy_std                    -0.833            0.694
```

**Psychometric Statistics:**
- **Cronbach's α:** 0.736 (Acceptable reliability)
- **KMO Sampling Adequacy:** 0.665 (Adequate)
- **Bartlett's Test:** χ² = 189.469, p < 0.001 (Significant)
- **Eigenvalue:** 1.497
- **Variance Explained:** 37.4%

### 1.5 IDEOLOGY Bundle

**Variables:** Prosocial_DV, ClassSystemJust_std, EconomicBelief_std, SupportEquality_std, Egalitarianism_std, SocialOrientation_std, HumanVS_std, RSDO_std

**Correlation Matrix:**
```
                       Prosocial_DV  ClassSystemJust_std  EconomicBelief_std  SupportEquality_std  Egalitarianism_std  SocialOrientation_std  HumanVS_std  RSDO_std
Prosocial_DV                  1.000               -0.086               0.038                0.116               0.075                  0.040       -0.012     0.065
ClassSystemJust_std          -0.086                1.000              -0.326               -0.652              -0.499                 -0.495        0.079    -0.560
EconomicBelief_std            0.038               -0.326               1.000                0.345               0.270                  0.258       -0.089     0.297
SupportEquality_std           0.116               -0.652               0.345                1.000               0.695                  0.463        0.166     0.627
Egalitarianism_std            0.075               -0.499               0.270                0.695               1.000                  0.575        0.107     0.897
SocialOrientation_std         0.040               -0.495               0.258                0.463               0.575                  1.000       -0.072     0.877
HumanVS_std                  -0.012                0.079              -0.089                0.166               0.107                 -0.072        1.000     0.024
RSDO_std                      0.065               -0.560               0.297                0.627               0.897                  0.877        0.024     1.000
```

**Factor Loadings (Two Factors Extracted):**
```
Variable                     IDEOLOGY_F1   IDEOLOGY_F2   Communality (h²)
Prosocial_DV                      -0.010         0.124            0.015
ClassSystemJust_std               -0.307        -0.448            0.295
EconomicBelief_std                 0.167         0.247            0.089
SupportEquality_std               -0.007         1.009            1.018
Egalitarianism_std                 0.631         0.262            0.467
SocialOrientation_std              0.884        -0.085            0.789
HumanVS_std                       -0.133         0.189            0.053
RSDO_std                           1.017         0.012            1.034
```

**Psychometric Statistics:**
- **Cronbach's α:** 0.390 (Poor reliability)
- **KMO Sampling Adequacy:** 0.741 (Good)
- **Bartlett's Test:** χ² = 10,410.918, p < 0.001 (Significant)
- **Eigenvalues:** F1 = 2.354, F2 = 1.406
- **Variance Explained:** F1 = 29.4%, F2 = 17.6%

---

## 2. DATA-DRIVEN EFA RESULTS

### 2.1 Overall Structure

**Variables Analyzed:** 27 (Prosocial_DV + 26 survey variables)  
**Factors Extracted:** 8 (eigenvalue > 1 criterion)  
**Total Variance Explained:** 45.5%

### 2.2 Factor Loadings Matrix (First 10 Variables)

```
Variable                     Factor1  Factor2  Factor3  Factor4  Factor5  Factor6  Factor7  Factor8
Prosocial_DV                  -0.098    0.137    0.017    0.188   -0.003   -0.109   -0.068    0.176
TotalAllowance_std            -0.080    0.048    0.015    0.122    0.041    0.091   -0.258    0.091
Age_std                       -0.090   -0.097    0.122    0.143    0.059   -0.196   -0.027   -0.037
SVO_type_std                   0.069   -0.048    0.071    0.037    0.065   -0.055    0.018    0.465
Prosocial_Motivation_std      -0.054    0.005    0.365    0.080   -0.085   -0.031    0.509    0.235
Agreeable_std                  0.091    0.069    0.824    0.054   -0.028    0.032    0.112    0.045
Integrity_Honesty_std         -0.053    0.077    0.492   -0.018    0.468    0.082   -0.141   -0.075
Kindness_std                  -0.024    0.045    0.395    0.197    0.324    0.065    0.237    0.065
Honesty_Humility_std           0.128   -0.055    0.659    0.034    0.062   -0.008   -0.119    0.062
NeuroticismBig5_std            0.145   -0.690   -0.061    0.054   -0.162    0.074    0.324   -0.064
```

### 2.3 Factor Eigenvalues and Variance

```
Factor    Eigenvalue    Variance Explained (%)
Factor1      2.426              9.0%
Factor2      1.909              7.1%
Factor3      1.878              7.0%
Factor4      1.463              5.4%
Factor5      1.410              5.2%
Factor6      1.188              4.4%
Factor7      1.076              4.0%
Factor8      0.905              3.4%
```

### 2.4 Highest Loading Variables per Factor

```
Factor1: RSDO_std (λ = 1.030) - Social Dominance
Factor2: SubHappy_std (λ = 0.933) - Well-being
Factor3: Agreeable_std (λ = 0.824) - Agreeableness
Factor4: ClassSystemJust_std (λ = -0.760) - System Justification
Factor5: ConscientiousnessBig5_std (λ = 0.761) - Conscientiousness
Factor6: Religious_std (λ = 0.743) - Religiosity
Factor7: Prosocial_Motivation_std (λ = 0.509) - Prosocial Motivation
Factor8: Dictator_1_std (λ = 0.677) - Dictator Game
```

---

## 3. MODEL PERFORMANCE COMPARISON

### 3.1 Predictive Performance

```
Approach                R²      Adj. R²    Key Finding
Theory-Driven EFA      0.117     0.067     PROSOCIAL factor significant (p = 0.008)
Data-Driven EFA        0.078     0.020     No significant personality factors
```

### 3.2 Reliability Summary

```
Bundle/Factor          Cronbach's α    Interpretation
PROSOCIAL                   0.548           Poor
HONESTY                     0.765           Acceptable  
BIGFIVE                     0.204           Poor
AFFECT                      0.736           Acceptable
IDEOLOGY                    0.390           Poor
```

---

## 4. METHODOLOGICAL INSIGHTS

### 4.1 Key Findings

1. **Theory-driven bundling outperformed data-driven clustering** in predictive validity
2. **Only HONESTY and AFFECT bundles achieved acceptable reliability** (α > 0.7)
3. **PROSOCIAL bundle had poor reliability** (α = 0.548) despite being the most predictive
4. **Income (TotalAllowance) consistently emerged as strongest predictor** across both approaches

### 4.2 Why DV Appears in Correlation Matrices

The dependent variable (Prosocial_DV) appears in all correlation matrices because we explicitly included it in each EFA bundle. This was our implementation of the professor's instruction to "include the DV" in factor analysis:

- **PROSOCIAL Bundle:** DV correlates weakly with prosocial measures (r = 0.052-0.159)
- **HONESTY Bundle:** DV has minimal correlation with honesty measures (r = 0.009-0.112)
- **AFFECT Bundle:** DV shows weak positive correlation with well-being (r = 0.015-0.091)


