# Vendor Selection Tables Restored

## Summary

Restored transparency tables to the Vendor Selection (Decision 8) results page that show:
1. **Vendor Score Breakdown Table** - Transparent calculation of composite scores
2. **Agent-Vendor Proximity Matrix Table** - Interactive display of proximity scores

## Tables Added

### 1. Vendor Score Breakdown Table

**Location**: After vendor attribute comparison charts in `render_vendor_selection()`

**Purpose**: Shows transparent calculation of how each vendor's composite score is computed.

**Columns** (18 total):
- Vendor
- Price ($) - Raw price
- Norm Price - Normalized [0,1] (inverted: lower price = higher score)
- Price Weight - Weight for price component
- Price Component - Price Weight × Norm Price
- Quality (1-5) - Raw quality rating
- Norm Quality - Normalized [0,1]
- Quality Weight - Weight for quality component
- Quality Component - Quality Weight × Norm Quality
- Sustainability (1-5) - Raw sustainability rating
- Norm Sustain - Normalized [0,1]
- Sustain Weight - Weight for sustainability component
- Sustain Component - Sustain Weight × Norm Sustain
- Avg Proximity - Average proximity across all agents
- Norm Proximity - Normalized [0,1]
- Proximity Weight - Weight for proximity component
- Proximity Component - Proximity Weight × Norm Proximity
- **Final Score** - Sum of all components

**Formula Displayed**:
```
Final Score = (Price Weight × Norm Price) + (Quality Weight × Norm Quality) + 
              (Proximity Weight × Norm Proximity) + (Sustainability Weight × Norm Sustainability)
```

**Notes**:
- Weights are averaged across all agents
- Proximity is averaged across all agents per vendor
- Price normalization is inverted (lower price gets higher score)

---

### 2. Agent-Vendor Proximity Matrix Table

**Location**: After vendor score breakdown table in `render_vendor_selection()`

**Purpose**: Display the proximity matrix showing each agent's proximity score to each vendor.

**Display Options**:
- **Default**: Shows first 20 agents (for manageable display)
- **Optional**: Checkbox to "Show all agents" for complete matrix

**Columns**:
- Agent ID
- Assigned Allowance Level
- Group_experiment
- Vendor 1 Proximity
- Vendor 2 Proximity
- ... (one column per vendor)

**Features**:
- Expandable section to keep UI clean
- Height-limited scrollable table
- Shows agent count and total proximity scores
- Excel download still available below the table

---

## Files Modified

**app/pages/results/visualizations/vendor_viz.py**:
- Lines 948-1018: Added Vendor Score Breakdown Table
- Lines 1020-1072: Enhanced Proximity Matrix with table display (not just download)

## Implementation Details

### Vendor Score Breakdown

The table uses:
- Existing `avg_proximity_per_vendor` dictionary (already calculated)
- Existing `vendors_data` from session state
- Average weights calculated from all agents' `vendor_choice_weights`
- Same normalization logic as `_calculate_vendor_score()` function

### Proximity Matrix Display

The table uses:
- Existing `proximity_matrix_data` build logic
- Interactive checkbox for all/sample view
- Expandable section to avoid overwhelming UI
- Consistent with existing Excel export structure

## Benefits

✅ **Transparency**: Users can see exactly how vendor scores are calculated  
✅ **Verification**: Can verify normalization and weight application  
✅ **Analysis**: Can understand why certain vendors were selected  
✅ **Proximity Insights**: Can see agent-vendor distance patterns  
✅ **Interactive**: Can toggle between sample and full matrix view  

## Usage

Run a simulation with multiple vendors (N ≥ 2) and view Decision 8 results:
1. Scroll to "Vendor Data & Selection Analysis" section
2. **NEW**: See "Vendor Score Breakdown" table showing all score components
3. **NEW**: Expand "View Proximity Matrix Table" to see agent-vendor proximity scores
4. Download buttons remain available for Excel export



