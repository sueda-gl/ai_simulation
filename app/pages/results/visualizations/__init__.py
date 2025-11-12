# app/pages/results/visualizations/__init__.py
"""
Visualization module for decision-specific render functions.
This module organizes all visualization functions by category.
"""

# Import all visualization functions from category modules
from .donation_viz import (
    render_donation_default,
    render_final_donation_rate
)

from .disclosure_viz import (
    render_disclose_income,
    render_disclose_documents
)

from .transaction_viz import (
    render_purchase_vs_bid,
    render_rejected_transaction_defaults,
    render_rejected_transaction_option,
    render_rejected_bid_value
)

from .vendor_viz import (
    render_vendor_choice_weights,
    render_vendor_selection
)

from .purchasing_viz import (
    render_purchasing_quantity,
    render_purchasing_frequency
)

from .bidding_viz import (
    render_bid_value
)

from .viz_helpers import (
    render_probability_controls,
    get_dynamic_description,
    _render_missing_visualization
)


# Registry mapping decision names to their visualization functions
DECISION_VISUALIZATIONS = {
    # Donation decisions
    'donation_default': render_donation_default,
    'donation_default_raw': render_donation_default,  # Same as donation_default
    'final_donation_rate': render_final_donation_rate,
    
    # Disclosure decisions
    'disclose_income': render_disclose_income,
    'disclose_documents': render_disclose_documents,
    
    # Transaction decisions
    'rejected_transaction_defaults': render_rejected_transaction_defaults,
    'purchase_vs_bid': render_purchase_vs_bid,
    'rejected_transaction_option': render_rejected_transaction_option,
    'rejected_bid_value': render_rejected_bid_value,
    
    # Vendor decisions
    'vendor_choice_weights': render_vendor_choice_weights,
    'vendor_selection': render_vendor_selection,
    
    # Purchasing decisions
    'purchasing_quantity': render_purchasing_quantity,
    'purchasing_frequency': render_purchasing_frequency,
    
    # Bidding decisions
    'bid_value': render_bid_value,
    
    # Note: Any decision not in this registry will automatically use render_generic_decision
}


# Export all symbols
__all__ = [
    # Donation
    'render_donation_default',
    'render_final_donation_rate',
    
    # Disclosure
    'render_disclose_income',
    'render_disclose_documents',
    
    # Transaction
    'render_purchase_vs_bid',
    'render_rejected_transaction_defaults',
    'render_rejected_transaction_option',
    'render_rejected_bid_value',
    
    # Vendor
    'render_vendor_choice_weights',
    'render_vendor_selection',
    
    # Purchasing
    'render_purchasing_quantity',
    'render_purchasing_frequency',
    
    # Bidding
    'render_bid_value',
    
    # Helpers
    'render_probability_controls',
    'get_dynamic_description',
    '_render_missing_visualization',
    
    # Registry
    'DECISION_VISUALIZATIONS',
]

