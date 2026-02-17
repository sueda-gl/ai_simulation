## 3. The Stochastic Component (σ) and Anchor Mix

After the regression model predicts a score (ŷ'_i), it proceeds to the "Anchor" and "Stochastic" options.

- **The Stochastic Component**: This is an optional randomness element that can be introduced by checking the checkbox and specifying the "σ Coefficient" on the slider. It is enabled by default for the research specification configuration and disabled permanently for the research baseline mode which is non-stochastic with no randomness. The two main differences between research specification and research baseline modes are: (1) a stochastic element which is disabled in the research baseline mode and (2) population sampling: if the number of specified agents n exceeds 280 for the research specification mode, bootstrap sampling is implemented, while for the research baseline mode, the first n agents are selected. The user can disable the stochastic element for research specification to compare these two modes, but the population sampling procedure will remain distinct for the two modes. The stochastic element is disabled by default for the copula mode and could be optionally enabled for comparison with the research specification mode. When enabled, it transforms the deterministic anchor into a probabilistic outcome.

  – **Mechanism**: The model makes a random draw from a Normal (Gaussian) distribution:
  Draw ~ N(μ, σ)

  – The **Mean** (μ) of this distribution is the **adjusted anchor value** calculated in the previous step.

  – The **Standard Deviation** (σ), which controls the amount of randomness, is user-configurable. The user can choose between two σ modes:

  **Mode 1: Uniformly (Overall σ)**

  One common σ for all agents:

  ```
  σ = Base σ_empirical × σ_Coefficient
  ```

  Where Base σ is derived from the variance in the original research data corresponding to the observed measure of prosocial behavior, i.e., TWT+Sospeso [=AW2+AX2]{Periods 1+2}, and the σ_Coefficient is the value defined by the user's slider. Setting the slider to 0 makes σ = 0, which effectively disables the stochastic component. Leaving it at 1 (the default) introduces the observed standard deviation. A value of 0.1 is typical for the σ_Coefficient.

  Base σ_empirical = sd(TWT+Sospeso) = 9.8995

  Examples:
  - σ_Coefficient = 1.0 → σ = 9.8995 (full empirical variability)
  - σ_Coefficient = 0.1 → σ = 0.98995 (reduced variability)
  - σ_Coefficient = 0.0 → σ = 0 (no stochastic component)

  **Mode 2: Quintiles (Per-Income-Level σ)**

  Different σ for different income groups, providing income-specific precision. The base σ for each quintile is computed from the standard deviation of TWT+Sospeso **within** that income quintile:

  | Income Level | Budget | Base σ |
  |-------------|--------|--------|
  | Level 1 | €12 | 5.705052 |
  | Level 2 | €32 | 3.069326 |
  | Level 3 | €72 | 3.532226 |
  | Level 4 | €128 | 12.219622 |
  | Level 5 | €200 | 16.854622 |

  Each level has its own coefficient slider (default = 1.0, range 0.0–2.0):

  ```
  σ(q) = Base σ_quintile(q) × Coefficient(q)
  ```

  This makes the draw more "personalized" — different income groups get different amounts of noise based on how much variability was actually observed in that group. For example, Level 2 (€32) had very low variability (σ = 3.07), while Level 5 (€200) had high variability (σ = 16.85).

  **Important**: Per-quintile σ values only apply to the **categorical** income specification. In continuous mode (or the continuous run of "Compare both"), the simulation always falls back to the overall σ (Base σ × σ_Coefficient), because level-specific sigmas are based on categorical budget levels and are not meaningful for continuous income. The note on the UI dynamically reflects the current overall coefficient value.

  **Scale conversion**: Because the donation default model operates on a 0–100 scale (after scaling observed and predicted values), the sigma is converted from the original 0–112 TWT+Sospeso scale to the 0–100 working scale before the draw:

  ```
  σ_0-100 = σ_raw × (100 / 112)
  ```

  If the stochastic component is disabled, the model selected by the user uses the deterministic adjusted anchor value as its output. This process enables nuanced exploration of randomness versus deterministic prediction of prosocial behavior, which influences the decision outcome.

- **Anchor Mix**: The anchor is a deterministic blend of the observed participant behavior as documented in the original data (based on personal and community donations made) and the prediction of its prosocial score from the regression model as noted above. The "Anchor Mix" slider directly controls these weights (w), with a default of w = 0.75 on a 0-1 scale:

  ```
  Anchor = w · Observed_scaled + (1 − w) · Predicted_scaled
  ```

  Both observed and predicted values are first scaled to a common 0–100 scale before combining.

- **Distribution Adjustment**: This "Distribution Adjustment" shift_value can be added by the user to the anchor. This allows a user to manually shift the distribution of donation rate for the whole population, thus adjusting the model's intercept.
