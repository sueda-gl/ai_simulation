# Phase C (revised): Bundle-wise EFA
### PROSOCIAL
Items used: SVO_type_std, Prosocial_Motivation_std, Dictator_1_std
N = 280 rows (complete-case within bundle)
KMO = 0.62, Bartlett p = 4.378e-13
Factors retained = 1

### HONESTY
Items used: Honesty_Humility_std, Integrity_Honesty_std, Kindness_std, Agreeable_std
N = 280 rows (complete-case within bundle)
KMO = 0.72, Bartlett p = 6.714e-59
Factors retained = 1

*BIGFIVE* KMO=0.57 (<0.6). Bundle skipped.
### AFFECT
Items used: PosAffect_std, LifeSatis_std, SubHappy_std
N = 280 rows (complete-case within bundle)
KMO = 0.66, Bartlett p = 3.3e-38
Factors retained = 1

### IDEOLOGY
Items used: ClassSystemJust_std, EconomicBelief_std, SupportEquality_std, Egalitarianism_std, SocialOrientation_std, HumanVS_std, RSDO_std
N = 280 rows (complete-case within bundle)
KMO = 0.74, Bartlett p = 0
Factors retained = 2


Outputs generated:

- efa_loadings_PROSOCIAL.csv
- efa_loadings_HONESTY.csv
- efa_loadings_BIGFIVE.csv
- efa_loadings_AFFECT.csv
- efa_loadings_IDEOLOGY.csv
- efa_factor_scores.csv (all bundles concatenated)
- analysis_table_with_multi_factors.csv (master with scores)