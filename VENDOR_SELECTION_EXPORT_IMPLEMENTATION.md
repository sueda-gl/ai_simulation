# Vendor Selection Export Implementation

## Summary
Enhanced the vendor selection results page with two new export capabilities:

### 1. Agent-Vendor Proximity Score Matrix Export
**Location:** `app/pages/results/visualizations/vendor_viz.py` (lines 509-568)

**Functionality:**
- Replaces the previous "Sample: Individual Agent-Vendor Proximity Scores (First 10 Agents)" display
- Provides downloadable Excel file with complete proximity matrix
- Shows each agent's proximity score to each vendor

**Excel Structure:**
- **Sheet Name:** "Agent-Vendor Proximity"
- **Columns:**
  - Agent ID
  - Assigned Allowance Level (if available)
  - Group_experiment (if available)
  - Vendor 1 Proximity
  - Vendor 2 Proximity
  - ... (one column per vendor)

**File Name:** `agent_vendor_proximity_matrix_YYYYMMDD_HHMMSS.xlsx`

---

### 2. Purchase Request-Level Data Export
**Location:** `app/pages/results/visualizations/vendor_viz.py` (lines 479-353)

**Functionality:**
- Provides comprehensive purchase request-level data
- Includes vendor attributes and transaction outcomes
- Multiple sheets organized by period

**Excel Structure:**

#### Main Sheet: "Total"
Contains all purchase requests across all periods

#### Additional Sheets: "Period 1", "Period 2", etc.
One sheet per period with requests from that period only

**Columns (17 total):**
1. **Transaction ID** - Unique identifier for the purchase request
2. **Agent ID** - Customer/agent identifier
3. **Assigned Allowance Level** - Income/allowance tier
4. **Group_experiment** - Experimental group assignment
5. **Customer Type** - Regular, Fixed, or Discount
6. **Request Date & Time** - Formatted as "Period X, Hour Y.Y"
7. **Period** - Period number (derived from timestamp)
8. **Selected Vendor** - Vendor chosen for this request
9. **Vendor Price** - Price offered by selected vendor
10. **Quality** - Vendor quality rating (1-5)
11. **Sustainability** - Vendor sustainability rating (1-5)
12. **Proximity** - Agent-vendor proximity score (0-100)
13. **Vendor Integrated Score** - Composite score calculated from weighted attributes
14. **Transaction Completed** - 0/1 indicator (or NaN if not tracked)
15. **Customer Paid Price** - Price paid without donation

**File Name:** `purchase_requests_detailed_YYYYMMDD_HHMMSS.xlsx`

---

## Implementation Details

### Helper Functions

#### `_build_purchase_request_export(df, vendors_data)`
**Purpose:** Extract and transform purchase request data from simulation results

**Process:**
1. Iterates through each agent in the results DataFrame
2. Extracts agent-level metadata (ID, allowance, group)
3. Loops through each purchase request for the agent
4. Retrieves vendor attributes from vendor lookup
5. Calculates vendor integrated score using weighted formula
6. Builds comprehensive record with all fields
7. Returns list of records ready for DataFrame conversion

**Key Features:**
- Handles missing vendor data gracefully
- Converts timestamp_hours to readable "Period X, Hour Y" format
- Calculates period automatically (24 hours per period)
- Capitalizes customer type for consistency
- Supports multiple field name variations for compatibility

#### `_calculate_vendor_score(vendor, weights, proximity, all_vendors)`
**Purpose:** Calculate vendor integrated composite score

**Formula:**
```
score = w_price × norm_price + w_quality × norm_quality + 
        w_proximity × norm_proximity + w_sustainability × norm_sustainability
```

**Normalization:**
- **Price:** Inverted min-max normalization (lower price = higher score)
  - `norm_price = 1 - ((price - min_price) / (max_price - min_price))`
- **Quality:** Linear scaling from [1,5] to [0,1]
  - `norm_quality = (quality - 1) / 4`
- **Sustainability:** Linear scaling from [1,5] to [0,1]
  - `norm_sustainability = (sustainability - 1) / 4`
- **Proximity:** Linear scaling from [0,100] to [0,1]
  - `norm_proximity = proximity / 100`

**Weights:** Obtained from agent's vendor_choice_weights (configured in Page 2 Overview)

---

## Data Sources

### From Simulation Results DataFrame:
- `agent_id` - Agent identifier
- `Assigned Allowance Level` - Income tier
- `Group_experiment` - Experimental group
- `purchase_requests` - List of purchase request dictionaries
- `vendor_proximity_scores` - Dict mapping vendor IDs to proximity scores
- `vendor_choice_weights` - Dict with weights for price, quality, proximity, sustainability

### From Purchase Request Dictionaries:
- `request_id` or `transactionID` - Request identifier
- `vendorID` - Selected vendor ID
- `timestamp_hours` - Request timestamp in hours
- `customer_type` - Customer category (discount, fixed, regular)
- `platformPrice` - "DISCOUNT", "FIXED", "PN", or "BID"
- `bid_value` - Bid amount or "N/A"
- Transaction completion fields (if available)
- Price paid fields (if available)

### From Vendor Data (session_state.vendors):
- `vendor_id` - Vendor identifier
- `price` - Vendor's offered price
- `quality` - Quality rating (1-5)
- `sustainability` - Sustainability rating (1-5)
- `quantity_offered` - Vendor capacity per period

---

## UI Integration

### Location in Results Page
Both exports appear in the **Vendor Selection** section of the Results page, after the vendor distribution visualization.

### User Experience
1. **Purchase Request Export:**
   - Appears immediately after vendor selection distribution chart
   - Shows summary info about number of requests and sheets
   - Download button: "📥 Download Purchase Requests Excel"

2. **Proximity Matrix Export:**
   - Appears in the "Vendor Data & Selection Analysis" section
   - Only shown when multiple vendors exist
   - Download button: "📊 Download Proximity Matrix Excel"

### Error Handling
- Checks for openpyxl package availability
- Gracefully handles missing vendor data
- Provides informative messages when data unavailable
- Shows NaN for fields not yet tracked in simulation

---

## Testing Recommendations

1. **Single Vendor Scenario:**
   - Test that exports work with 1 vendor
   - Verify vendor attributes are retrieved correctly

2. **Multi-Vendor Scenario:**
   - Test with 3-5 vendors
   - Verify proximity matrix has correct dimensions
   - Check that vendor integrated scores are calculated correctly

3. **Multi-Period Scenario:**
   - Run simulation with multiple periods
   - Verify separate sheets are created for each period
   - Confirm "Total" sheet contains all requests

4. **Edge Cases:**
   - Agents with 0 purchase requests
   - Missing vendor data
   - Missing proximity scores
   - Different customer types (discount, fixed, regular)

---

## Future Enhancements

1. **Transaction Completion Tracking:**
   - Add actual transaction outcome tracking in simulation
   - Update `transactionCompleted` field in purchase requests
   - Track rejected vs. accepted transactions

2. **Price Paid Tracking:**
   - Implement actual price calculation with donations
   - Track final price paid by customer
   - Include donation amounts separately

3. **Additional Metrics:**
   - Request processing time
   - Vendor capacity remaining
   - Alternative vendor rankings
   - Bid acceptance/rejection outcomes

4. **Export Customization:**
   - Allow user to select which columns to include
   - Add filtering by customer type or period
   - Include summary statistics sheet
   - Add charts/visualizations in Excel

---

## Files Modified

- `app/pages/results/visualizations/vendor_viz.py`
  - Added `_build_purchase_request_export()` function
  - Added `_calculate_vendor_score()` function
  - Modified `render_vendor_selection()` to include exports
  - Replaced sample proximity display with matrix download
  - Added numpy import for NaN handling

## Dependencies

- `pandas` - DataFrame operations
- `openpyxl` - Excel file creation
- `streamlit` - UI components
- `numpy` - NaN handling and numerical operations
- `datetime` - Timestamp formatting
- `io.BytesIO` - In-memory file buffer

## File Size Considerations

**Proximity Matrix:**
- Small: 100 agents × 3 vendors = 300 scores (~10 KB)
- Medium: 500 agents × 5 vendors = 2,500 scores (~50 KB)
- Large: 1000 agents × 10 vendors = 10,000 scores (~200 KB)

**Purchase Requests:**
- Small: 100 agents × 5 requests × 3 periods = 1,500 rows (~100 KB)
- Medium: 500 agents × 7 requests × 5 periods = 17,500 rows (~1 MB)
- Large: 1000 agents × 10 requests × 10 periods = 100,000 rows (~5 MB)

All file sizes are well within Excel's limits (1 million rows per sheet).

