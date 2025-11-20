# Page 2 Parameter Persistence Issue

## Problem Identified

Page 2 (Decision Parameters) has the same issue as Page 1 had. In the `donation_default` tab, widgets read their `value` parameter from session state variables like `st.session_state.sigma_coefficient`, but the widget's `key` parameter uses a different name like `tab_sigma_coefficient`.

### Example of the Problem:

```python
sigma_coefficient = st.slider(
    "σ Coefficient (multiplier)",
    value=st.session_state.sigma_coefficient,  # ← Reads from sigma_coefficient
    key="tab_sigma_coefficient"  # ← But widget key is different!
)
st.session_state.sigma_coefficient = sigma_coefficient  # Sync back
```

When you navigate away and back:
1. Widget reads from `st.session_state.sigma_coefficient` 
2. If that value got reset during navigation, your change is lost
3. Even though `tab_sigma_coefficient` key still has your value!

## Affected Widgets in donation_default Tab

1. `sigma_in_copula` checkbox (3 versions for different modes)
2. `sigma_in_research` checkbox (2 versions)
3. `sigma_coefficient` slider (3 versions for different modes)
4. `anchor_observed_weight` slider

## Solution

Make widgets read from their own keys, just like we fixed in Page 1.




