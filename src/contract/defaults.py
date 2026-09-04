# src/contract/defaults.py
"""The default values and default descriptions of the 13 decisions.

Pure data, moved VERBATIM out of ``app/pages/decision_execution.py``: the same
dicts, the same keys, the same comments.  They belong to the run contract - the
seam (``app.seam.build_plan``) and the screens both need them - so they live
here, where nothing imports Streamlit and nothing has to reach back into the
app package.

``app.pages.decision_execution`` re-exports both names, so every historical
``from app.pages.decision_execution import DEFAULT_DECISION_VALUES`` keeps
working and there is still exactly one definition.
"""

# Default values for unselected decisions
DEFAULT_DECISION_VALUES = {
    "donation_default": 0.10,  # 10%
    "disclose_income": {
        "type": "random_probability",
        "probability_y": 0.5,  # 50% chance of Y (disclosing)
        "options": ["Y", "N"],
        "description": "Probability of disclosing income for Fixed status"
    },
    "disclose_documents": {
        "type": "random_probability",
        "probability_y": 0.5,  # 50% chance of Y (disclosing)
        "options": ["Y", "N"],
        "description": "Probability of disclosing documents (applies only to agents qualified for discount: income < threshold)"
    },
    "rejected_transaction_defaults": {
        "type": "prioritized_selection",
        "priority_template": ["forgo_transaction"],  # Default: all agents use Option 5 only
        "options": [
            ("higher_price_category", "Option 1: Purchase from another (higher) price category of the same vendor"),
            ("lower_pn_vendor", "Option 2: Purchase from another vendor at PN price which is lower than the PN price of the current vendor"), 
            ("current_vendor_pn", "Option 3: Purchase from the current vendor at PN price"),
            ("place_bid", "Option 4: Place a bid for the current vendor in the current period (rejected fixed) or next period (rejected bids/discount)"),
            ("forgo_transaction", "Option 5: Forgo the purchase request")
        ],
        "description": "Each agent gets a prioritized list. If Option 5 is included, it must be last."
    },
    "vendor_choice_weights": {
        "type": "checkbox_selection",
        "default_selection": ["price", "quality", "proximity", "sustainability"],
        "parameters": {
            "price": {"name": "Price", "description": "the product price offered to the customer"},
            "quality": {"name": "Quality", "description": "product quality based on customer ratings"},
            "proximity": {"name": "Proximity", "description": "the proximity of vendor to customer"},
            "sustainability": {"name": "Sustainability", "description": "vendor sustainability rating"}
        }
    },
    "purchasing_quantity": "RANDOM_WITHIN_LIMIT",  # Random within purchasing limit
    "purchasing_frequency": "CALCULATED",  # Consumption quantity / Number of Periods
    "vendor_selection": "deterministic",  # Deterministic based on highest weighted vendor-product score
    "purchase_vs_bid": {
        "type": "random_probability",
        "probability_y": 0.5,  # 50% chance of Purchase Now (vs bid)
        "options": ["Purchase Now", "bid"],
        "description": "Probability of Purchase Now vs bidding (applies only to REGULAR customers - those who did not disclose income)"
    },
    "bid_value": "RANDOM_WITHIN_RANGE",  # Random within bidding price range
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
    },
    "rejected_bid_value": "NA",  # Not relevant given Option 5
    "final_donation_rate": 0.10  # Keep default 10%
}

# Description text for display purposes
DEFAULT_DECISION_DESCRIPTIONS = {
    "donation_default": "10%",
    "disclose_income": "configurable probability Y/N (default 50% each)", 
    "disclose_documents": "configurable probability Y/N (applies only to agents with income < discount threshold, default 50% each)",
    "rejected_transaction_defaults": "Selected option for handling rejected transactions will be applied to all agents",
    "vendor_choice_weights": "equal weight distribution among selected parameters (Price, Quality, Proximity, Sustainability)",
    "purchasing_quantity": "random within purchasing limit",
    "purchasing_frequency": "Consumption quantity divided by Number of Periods",
    "vendor_selection": "deterministic based on highest weighted vendor-product score",
    "purchase_vs_bid": "configurable probability Purchase Now/bid for REGULAR customers only (default 50% each)",
    "bid_value": "random within bidding price range (only for REGULAR customers who chose to bid)",
    "rejected_transaction_option": "Selected specific option for transaction rejection handling will be used",
    "rejected_bid_value": "Default handling for rejected bid values will be applied",
    "final_donation_rate": "Default donation rate will be maintained"
}

__all__ = ["DEFAULT_DECISION_VALUES", "DEFAULT_DECISION_DESCRIPTIONS"]
