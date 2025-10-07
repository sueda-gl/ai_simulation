# Rejected Transaction Options & Vendor Selection Updates

## Summary
Updated rejected transaction options throughout the codebase to match the actual platform options, and corrected vendor selection default behavior wording.

## Date
October 7, 2025

---

## Changes Made

### 1. Rejected Transaction Options Updated

#### Previous Options (Incorrect)
```
Option 1: Reduce Bid Amount
Option 2: Switch to Different Vendor
Option 3: Choose Different Product
Option 4: Wait and Retry Later
Option 5: Forgo Transaction
```

#### New Options (Correct - Matches Platform)
```
Option 1: Purchase from another (higher) price category of the same vendor
Option 2: Purchase from another vendor at PN price which is lower than the PN price of the current vendor
Option 3: Purchase from the current vendor at PN price
Option 4: Place a bid for the current vendor in the current period (rejected fixed) or next period (rejected bids/discount)
Option 5: Forgo the purchase request
```

#### Internal Keys Updated
- `reduce_bid` → `higher_price_category`
- `switch_vendor` → `lower_pn_vendor`
- `switch_product` → `current_vendor_pn`
- `retry_later` → `place_bid`
- `forgo_transaction` → `forgo_transaction` (unchanged)

---

### 2. Vendor Selection Wording Updated

#### Previous Wording
```
"deterministic based on vendor choice weights"
```

#### New Wording
```
"deterministic based on highest weighted vendor-product score"
```

This better reflects the actual algorithm that calculates a weighted score for each vendor-product combination and selects the highest scoring option.

---

## Files Modified

### 1. `/Users/suedagul/<sdg/app/pages/decision_execution.py`

**Lines 112-121**: Updated `rejected_transaction_defaults` options
```python
"rejected_transaction_defaults": {
    "type": "radio_selection",
    "default_option": "forgo_transaction",
    "options": [
        ("higher_price_category", "Option 1: Purchase from another (higher) price category of the same vendor"),
        ("lower_pn_vendor", "Option 2: Purchase from another vendor at PN price which is lower than the PN price of the current vendor"), 
        ("current_vendor_pn", "Option 3: Purchase from the current vendor at PN price"),
        ("place_bid", "Option 4: Place a bid for the current vendor in the current period (rejected fixed) or next period (rejected bids/discount)"),
        ("forgo_transaction", "Option 5: Forgo the purchase request")
    ]
}
```

**Lines 143-152**: Updated `rejected_transaction_option` options
```python
"rejected_transaction_option": {
    "type": "radio_selection",
    "default_option": "forgo_transaction", 
    "options": [
        ("higher_price_category", "Option 1: Purchase from another (higher) price category of the same vendor"),
        ("lower_pn_vendor", "Option 2: Purchase from another vendor at PN price which is lower than the PN price of the current vendor"),
        ("current_vendor_pn", "Option 3: Purchase from the current vendor at PN price"), 
        ("place_bid", "Option 4: Place a bid for the current vendor in the current period (rejected fixed) or next period (rejected bids/discount)"),
        ("forgo_transaction", "Option 5: Forgo the purchase request")
    ]
}
```

**Line 135**: Updated comment
```python
"vendor_selection": "deterministic",  # Deterministic based on highest weighted vendor-product score
```

**Line 167**: Updated description
```python
"vendor_selection": "deterministic based on highest weighted vendor-product score",
```

### 2. `/Users/suedagul/<sdg/app/pages/results/decision_visualizations.py`

**Lines 325-330**: Updated `render_rejected_transaction_defaults()` options
```python
options = [
    ("higher_price_category", "Option 1: Purchase from another (higher) price category of the same vendor"),
    ("lower_pn_vendor", "Option 2: Purchase from another vendor at PN price which is lower than the PN price of the current vendor"),
    ("current_vendor_pn", "Option 3: Purchase from the current vendor at PN price"), 
    ("place_bid", "Option 4: Place a bid for the current vendor in the current period (rejected fixed) or next period (rejected bids/discount)"),
    ("forgo_transaction", "Option 5: Forgo the purchase request")
]
```

**Lines 739-743**: Updated `render_rejected_transaction_option()` options
```python
options = [
    ("higher_price_category", "Option 1: Purchase from another (higher) price category of the same vendor"),
    ("lower_pn_vendor", "Option 2: Purchase from another vendor at PN price which is lower than the PN price of the current vendor"),
    ("current_vendor_pn", "Option 3: Purchase from the current vendor at PN price"), 
    ("place_bid", "Option 4: Place a bid for the current vendor in the current period (rejected fixed) or next period (rejected bids/discount)"),
    ("forgo_transaction", "Option 5: Forgo the purchase request")
]
```

### 3. `/Users/suedagul/<sdg/DEFAULT_DECISIONS_FEATURE.md`

**Lines 94-98**: Updated documentation example
```markdown
○ Option 1: Purchase from another (higher) price category of the same vendor
○ Option 2: Purchase from another vendor at PN price which is lower than the PN price of the current vendor
○ Option 3: Purchase from the current vendor at PN price
○ Option 4: Place a bid for the current vendor in the current period (rejected fixed) or next period (rejected bids/discount)
● Option 5: Forgo the purchase request
```

### 4. `/Users/suedagul/<sdg/app/pages/decision_tabs/default_config.py`

**Line 260**: Updated vendor selection description in overview page
```python
"deterministic": "Deterministic selection based on highest weighted vendor-product score",
```

This appears in the Overview tab when `vendor_selection` is shown as an unselected (default) decision.

---

## Impact Assessment

### User Interface
- ✅ Radio buttons in Overview tab now show correct platform options
- ✅ Results page visualizations display correct option names
- ✅ Option descriptions are more descriptive and accurate
- ✅ Vendor selection description is more technically accurate

### Data Consistency
- ✅ Internal keys updated but `forgo_transaction` remains the same (backward compatible for default)
- ✅ Session state keys will automatically use new internal keys
- ✅ Radio button selections will map to correct option values

### Backend Compatibility
- ⚠️ **Important**: Backend simulation logic may need updates to handle new option keys
- ⚠️ Need to verify that `rejected_transaction_defaults.py` and `rejected_transaction_option.py` can process the new keys
- ⚠️ The options represent different behaviors, so backend implementation must be updated accordingly

### Documentation
- ✅ Documentation updated to reflect correct platform options
- ✅ Examples show accurate option text

---

## Testing Recommendations

### 1. UI Testing
- [ ] Navigate to Page 2 → Overview tab
- [ ] Verify rejected transaction options show new text
- [ ] Select each option and verify it's saved correctly
- [ ] Run simulation with different option selections
- [ ] Verify Results page shows correct option names

### 2. Backend Testing
- [ ] Verify backend can process new option keys:
  - `higher_price_category`
  - `lower_pn_vendor`
  - `current_vendor_pn`
  - `place_bid`
  - `forgo_transaction`
- [ ] Test each option's actual behavior in simulation
- [ ] Verify option logic matches the descriptions

### 3. Regression Testing
- [ ] Test with previously saved configurations (if any)
- [ ] Verify default behavior (Option 5) still works
- [ ] Test complete simulation with all defaults
- [ ] Test complete simulation with custom option selections

---

## Next Steps (If Backend Implementation Needed)

If the backend simulation logic doesn't already implement these specific behaviors, the following files may need updates:

1. **`src/decisions/rejected_transaction_defaults.py`**
   - Update to handle new option keys
   - Implement logic for each option

2. **`src/decisions/rejected_transaction_option.py`**
   - Update to handle new option keys
   - Implement logic for each option

3. **Orchestrator files** (if they handle rejected transactions):
   - `src/orchestrator.py`
   - `src/orchestrator_baseline.py`
   - `src/orchestrator_doc_mode.py`

4. **Add option behavior implementation**:
   - Option 1: Logic to find higher price category of same vendor
   - Option 2: Logic to find different vendor with lower PN price
   - Option 3: Logic to switch to PN price for current vendor
   - Option 4: Logic to place bid (considering transaction type)
   - Option 5: Forgo transaction (already implemented)

---

## Validation

- ✅ No linter errors introduced
- ✅ All files updated consistently
- ✅ Backward compatibility maintained for default option
- ✅ Documentation updated
- ✅ Option descriptions match platform specifications

---

## Notes

### Option 4 Complexity
Option 4 has conditional behavior:
- **Rejected fixed transactions**: Place bid in **current period**
- **Rejected bids/discount transactions**: Place bid in **next period**

This conditional logic should be implemented in the backend to properly handle different rejection scenarios.

### Option 1 & 2 Requirements
These options require:
- **Price category system**: Agent must be able to identify and navigate price categories
- **Vendor comparison**: Agent must be able to compare PN prices across vendors
- **Product availability**: System must check if alternatives are available

Make sure these capabilities exist in the backend implementation.

