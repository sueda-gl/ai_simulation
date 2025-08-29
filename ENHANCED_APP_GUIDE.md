# Enhanced AI Agent Simulation - Two-Page Interface Guide

## Overview

The enhanced simulation app (`app_enhanced.py`) implements a two-page interface that allows users to configure comprehensive simulation parameters before running the AI agent simulation.

## How to Run

```bash
streamlit run app_enhanced.py
```

## Page Structure

### Page 1: Common Simulation Parameters

This page contains all the general parameters that apply to the entire simulation:

#### 1. Time Parameters
- **Number of Periods**: Number of periods for simulation run (1-100)
- **Duration per Period**: Duration of each period in hours (converted to seconds for simulation)

#### 2. Market Parameters
- **Number of Vendors**: Number of vendors operating on the platform (1-50)
- **Average Market Price**: The average market price of the product
- **Min/Max Vendor Price**: The price range for vendor products

#### 3. Product Offering
- **Products per Vendor (NV)**: Number of products offered by each vendor at period start
- **Carryover**: Whether unsold products carry over to the next period
- **Bidding Percentage (bp)**: Proportion of products available for bidding (0-100%)
  - Number of auction products = bp × NV

#### 4. Pricing Parameters
- **Platform Markup (m)**: Platform's markup on vendor prices
  - Customer Price = (1+m) × Vendor Price
- **Price Range (r)**: Range for Purchase Now and Minimum Bid prices
  - Purchase Now Price = (1+r) × Customer Price
  - Minimum Bid Price = (1-r) × Customer Price
- **Price Grid (g)**: Number of price categories (must be odd, 3-21)

#### 5. Income Distribution
- **Distribution Type**: Choose from lognormal, pareto, or weibull
- **Income Range**: Minimum and maximum income values
- **Average/Median Income**: Central tendency of the distribution

#### 6. Income Categories
- **Discount Income Categories (NDIC)**: Number of discount categories (1-10)
- **Fixed Income Categories (NFIC)**: Number of fixed categories (1-10)

#### 7. Vendor Price Configuration
- **Random Generation**: Prices generated within min/max range with specified average
- **File Upload**: Upload a CSV with vendor_id and price columns

#### 8. Consumption Limits
- Set consumption limits per product for each income category per period

### Page 2: Decision-Specific Parameters

This page allows configuration of individual decision modules:

#### Decision Selection
- Select which decisions to include in the simulation
- Option to select all decisions at once

#### Decision-Specific Settings
For each selected decision, you can configure:

**Donation Default** (if selected):
- Anchor Weights: Balance between observed and predicted prosocial scores
- Income Mode: Categorical or continuous specification
- Stochastic Parameters: Standard deviation (σ) for random draws
- Copula Mode Options: Whether to add stochastic component in copula mode

**Other Decisions**: Currently show placeholder status or existing configuration

#### Additional Options
- **Number of Agents**: Synthetic agents to generate (10-50,000)
- **Simulation Mode**: Single Run or Monte-Carlo Study
- **Random Seed**: For reproducible results
- **Save Results**: Option to save outputs to files

### Results Page

After configuration, the simulation runs and displays:

1. **Parameter Summary**: Expandable view of all configured parameters
2. **Simulation Overview**: Basic statistics about the run
3. **Decision Results**: Tabbed interface showing results for each decision
4. **Export Options**: Download results as CSV and parameters as JSON

## Key Features

### 1. Parameter Validation
- Price grid automatically adjusted to be odd
- Min/max price constraints enforced
- Income range validation

### 2. Real-time Calculations
- Duration converted from hours to seconds
- Number of auction products calculated from bidding percentage
- Example pricing shown based on current parameters

### 3. State Persistence
- All parameters saved in session state
- Navigation between pages preserves settings
- Can go back to modify parameters before running

### 4. Enhanced Data Models
The app uses dataclasses to manage parameters:

```python
@dataclass
class SimulationParameters:
    periods: int = 1
    duration_hours: float = 1.0
    num_vendors: int = 5
    market_price: float = 10.0
    # ... other parameters
    
    def get_purchase_now_price(self, base_price: float) -> float:
        """Calculate Purchase Now price from base price"""
        customer_price = base_price * (1 + self.platform_markup)
        return customer_price * (1 + self.price_range)
```

## Integration Notes

Currently, the enhanced interface connects to the existing simulation engine. The full integration of new parameters (vendor competition, bidding mechanics, pricing grids, etc.) is marked for future development.

### What's Working Now:
- Two-page parameter configuration interface
- Parameter validation and calculations
- Integration with existing decision modules
- Results visualization and export

### Future Enhancements:
- Full market dynamics simulation
- Vendor competition mechanics
- Bidding and auction processes
- Period-based consumption tracking
- Income category-specific behaviors

## Example Workflow

1. **Start the App**: `streamlit run app_enhanced.py`

2. **Configure Common Parameters** (Page 1):
   - Set 5 periods, 2 hours each
   - Configure 10 vendors with $10 average price
   - Set 50% products for bidding
   - Choose lognormal income distribution

3. **Configure Decisions** (Page 2):
   - Select donation_default and vendor_selection
   - Adjust donation parameters
   - Set 1000 agents
   - Choose single run mode

4. **Run Simulation**:
   - Click "Run Simulation"
   - View results across different tabs
   - Download results and parameters

## File Outputs

When "Save Results" is enabled:
- `outputs/enhanced_params_[timestamp].json`: All configuration parameters
- `outputs/enhanced_results_[timestamp].parquet`: Simulation results

## Tips

1. **Start Simple**: Begin with default values and adjust gradually
2. **Use Tooltips**: Hover over parameter labels for detailed explanations
3. **Check Calculations**: Review the example pricing to ensure parameters are reasonable
4. **Save Parameters**: Export parameter JSON for reproducibility
5. **Compare Runs**: Use different seeds to test variability

## Troubleshooting

- **Price Grid Error**: Ensure the value is odd (3, 5, 7, 9, 11, etc.)
- **Navigation Issues**: Use the navigation buttons, not browser back
- **Large Simulations**: Start with fewer agents and periods for testing
