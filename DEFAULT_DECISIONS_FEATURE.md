# Default Decisions Pre-Configuration Feature

## 📋 Overview

This document describes the new feature that allows users to **view and configure default decision parameters BEFORE running simulations**, implemented in the Page 2 Overview tab.

## 🎯 Problem Solved

**Before**: 
- Users could only adjust default decision parameters AFTER running a simulation (in the Results page)
- No visibility into what default values would be used for unselected decisions
- Had to run simulation → see results → adjust → re-run

**After**:
- Users can see and configure ALL default parameters in the Overview tab BEFORE running
- Full transparency into what values will be used for unselected decisions
- Configure once, run simulation with desired defaults immediately

## 🏗️ Architecture

### New Files Created

1. **`app/pages/decision_tabs/default_config.py`**
   - Main UI module for rendering default decision configuration
   - Contains specialized renderers for each decision type
   - Handles session state initialization and reset functionality

### Modified Files

1. **`app/pages/page2_decisions.py`**
   - Added import for `render_default_decisions_config`
   - Integrated default config section into Overview tab (before Run button)
   - Updated caption text to reference configured defaults

2. **`app/pages/decision_execution.py`**
   - Updated `get_actual_default_value()` to support priority order:
     1. Pre-configured defaults from Overview tab (`{decision}_default_*`)
     2. Post-simulation adjustments from Results page (`{decision}_*`)
     3. Hard-coded defaults from `DEFAULT_DECISION_VALUES`

## 🎨 User Interface

### Location
**Page 2 → Overview Tab → "Configure Default Decision Parameters" section**

Appears between:
- Selected donation configuration display (if applicable)
- "Run Complete Simulation" button

### Organization

Default decisions are grouped by category in expandable sections:

1. **📄 Disclosure Decisions**
   - `disclose_income`
   - `disclose_documents`

2. **💳 Transaction Decisions**
   - `purchase_vs_bid`
   - `rejected_transaction_defaults`
   - `rejected_transaction_option`
   - `rejected_bid_value`

3. **🏪 Vendor & Product Decisions**
   - `vendor_choice_weights`
   - `vendor_selection`

4. **🛒 Consumption Decisions**
   - `consumption_quantity`
   - `consumption_frequency`

5. **💰 Donation & Bidding Decisions**
   - `donation_default`
   - `final_donation_rate`
   - `bid_value`

### UI Components by Decision Type

#### 1. **Probability Decisions** (Y/N, purchase/bid)
- **Decisions**: `disclose_income`, `disclose_documents`, `purchase_vs_bid`
- **UI**: Slider (0-100%) with ratio display and default indicator
- **Example**:
  ```
  P(Y) - Probability of Yes     [====●====] 50%
  Ratio: 50% : 50%               Default: 50%
  Y : N                          ✓ Default
  ```

#### 2. **Radio Selection** (Multiple options)
- **Decisions**: `rejected_transaction_defaults`, `rejected_transaction_option`
- **UI**: Radio buttons with selected option display
- **Example**:
  ```
  ○ Option 1: Purchase from another (higher) price category of the same vendor
  ○ Option 2: Purchase from another vendor at PN price which is lower than the PN price of the current vendor
  ○ Option 3: Purchase from the current vendor at PN price
  ○ Option 4: Place a bid for the current vendor in the current period (rejected fixed) or next period (rejected bids/discount)
  ● Option 5: Forgo the purchase request
  
  Selected: Option 5
  ✓ Default
  ```

#### 3. **Checkbox Selection** (Parameter weights)
- **Decisions**: `vendor_choice_weights`
- **UI**: Multiple checkboxes with weight calculation display
- **Example**:
  ```
  ☑ Price - Cost of the product/service
  ☑ Quality - Quality rating and reviews
  ☑ Proximity - Distance and convenience
  ☑ Sustainability - Environmental impact
  
  Parameters: 4/4
  Weight Each: 25%
  ✓ Default
  ```

#### 4. **Numeric Values** (Percentages, rates)
- **Decisions**: `donation_default`, `final_donation_rate`
- **UI**: Slider (0-100%) with percentage display
- **Example**:
  ```
  Default Value    [===●=======] 0.10
  Percentage: 10%
  
  Current: 0.10
  ✓ Default
  ```

#### 5. **Computed/Placeholder** (No configuration needed)
- **Decisions**: `consumption_quantity`, `bid_value`, `vendor_selection`, etc.
- **UI**: Info box explaining automatic computation
- **Example**:
  ```
  ℹ️ Computed During Simulation
  Random value within consumption limit (computed per agent 
  based on income category)
  
  💡 No pre-configuration needed - this value is 
  automatically calculated
  ```

## 🔄 Session State Keys

### Naming Convention

- **Pre-configured defaults** (Overview tab): `{decision_name}_default_*`
  - `disclose_income_default_probability_y`
  - `rejected_transaction_defaults_default_selection`
  - `vendor_choice_weights_default_params`
  - `donation_default_default_value`

- **Post-simulation adjustments** (Results page): `{decision_name}_*`
  - `disclose_income_probability_y`
  - `rejected_transaction_defaults_option`
  - `vendor_choice_weights_selection`
  - `donation_default_config`

### Priority Order

`get_actual_default_value()` checks in order:
1. **Pre-configured** (Overview tab) - Highest priority
2. **Post-simulation** (Results page) - Medium priority
3. **Hard-coded** (DEFAULT_DECISION_VALUES) - Fallback

This ensures:
- Users can set defaults before running
- Post-simulation tweaks still work
- System always has a fallback

## ✨ Features

### 1. **Visual Indicators**
- ✓ **Default**: Parameter is using system default value
- ⚙️ **Modified**: Parameter has been changed from default

### 2. **Reset Functionality**
- "🔄 Reset All Defaults" button
- Clears all pre-configured defaults for unselected decisions
- Reverts to system defaults

### 3. **Smart Display**
- Only shows unselected decisions (those not configured on individual tabs)
- If all decisions are selected: Shows "✅ All decisions are selected for custom configuration"
- Updates dynamically as decisions are selected/deselected

### 4. **Contextual Help**
- Each parameter has descriptive help text
- Explains what the parameter controls
- Shows current vs. default values

## 🔧 Technical Details

### Key Functions

#### `render_default_decisions_config(selected_decisions, all_decisions)`
Main entry point - renders the entire default config section

#### `render_decision_default_config(decision_name)`
Routes to appropriate renderer based on decision type

#### Specialized Renderers
- `render_probability_default_config()` - For Y/N and purchase/bid decisions
- `render_radio_default_config()` - For multi-option selections
- `render_checkbox_default_config()` - For parameter weight selections
- `render_numeric_default_config()` - For numeric values
- `render_placeholder_default_config()` - For computed values

#### `reset_all_default_parameters(unselected_decisions)`
Clears all pre-configured defaults and checkbox states

## 🎬 User Flow

### Complete Workflow

1. **Navigate to Page 2**
2. **Select decisions** for custom configuration
3. **Go to Overview tab**
4. **View default parameters** for unselected decisions (organized by category)
5. **Configure defaults** as needed:
   - Adjust probabilities with sliders
   - Select options with radio buttons
   - Choose parameters with checkboxes
   - Set numeric values
6. **Review configuration**:
   - See visual indicators (Default/Modified)
   - Check all settings are correct
7. **Click "Run Complete Simulation"**
8. Simulation runs with:
   - Custom parameters for selected decisions
   - Configured defaults for unselected decisions

### Example Scenario

**User wants to run complete simulation with only donation_default custom:**

1. On Page 2, select only `donation_default`
2. Configure donation parameters in its tab
3. Go to Overview tab
4. See 12 other decisions with default configurations
5. Adjust defaults as needed:
   - Set `disclose_income` probability to 70%
   - Change `rejected_transaction_defaults` to "Option 1: Reduce Bid"
   - Keep `vendor_choice_weights` at default (all 4 parameters)
6. Click "Run Complete Simulation"
7. Results show donation_default with custom params + 12 decisions with configured defaults

## 🚀 Benefits

✅ **Pre-simulation Visibility**: See all defaults before running
✅ **Full Control**: Configure any default parameter
✅ **Organized Interface**: Grouped by category, expandable sections
✅ **Smart Defaults**: System provides sensible defaults
✅ **Visual Feedback**: Clear indicators for default vs. modified
✅ **Easy Reset**: One-click reset to system defaults
✅ **Backward Compatible**: Existing post-simulation adjustments still work
✅ **Efficient**: Configure once, run multiple simulations

## 📝 Usage Tips

1. **Start with defaults**: System defaults are sensible - only change what you need
2. **Use categories**: Expand only the categories you want to configure
3. **Check indicators**: Look for ⚙️ Modified to see what you've changed
4. **Reset if needed**: Use "Reset All Defaults" to start fresh
5. **Combine approaches**: 
   - Use custom tabs for complex decisions requiring detailed configuration
   - Use default config for simple decisions that just need minor tweaks

## 🔮 Future Enhancements

Potential improvements:
1. **Preset profiles**: Save/load common default configurations
2. **Bulk operations**: Set multiple probabilities to same value
3. **Import/Export**: Save default configs to file
4. **Templates**: Pre-configured defaults for common scenarios
5. **Validation**: Warn if unusual parameter combinations detected
6. **History**: Track default parameter changes over time

## 📚 Related Documentation

- See `app/pages/decision_execution.py` for `DEFAULT_DECISION_VALUES` structure
- See `app/pages/results/main_results.py` for post-simulation adjustment UI
- See individual decision modules in `src/decisions/` for implementation details

---

**Implementation Date**: 2025-10-01  
**Version**: 1.0  
**Status**: ✅ Complete and tested


