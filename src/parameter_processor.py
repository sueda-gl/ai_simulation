# src/parameter_processor.py
"""
Parameter Processor: Transforms global simulation parameters into decision-specific contexts
This provides a scalable way to propagate global parameters to individual decisions
"""

import yaml
from pathlib import Path
from typing import Dict, Any, Optional
import numpy as np

class ParameterProcessor:
    """Processes and transforms global parameters for decision modules"""
    
    def __init__(self):
        # Load parameter influence mapping
        influences_path = Path(__file__).resolve().parents[1] / "config" / "parameter_influences.yaml"
        with open(influences_path, 'r') as f:
            self.influences = yaml.safe_load(f)['parameter_influences']
    
    def get_decision_context(self, decision_name: str, simulation_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Transform global simulation parameters into decision-specific context
        
        Args:
            decision_name: Name of the decision module
            simulation_config: Global simulation configuration
            
        Returns:
            Decision-specific context derived from global parameters
        """
        if decision_name not in self.influences:
            return {}
        
        decision_info = self.influences[decision_name]
        affected_params = decision_info.get('affected_by', [])
        influence_type = decision_info.get('influence_type', 'unknown')
        
        # Extract relevant parameters
        sim_params = simulation_config.get('simulation', {})
        context = {
            'influence_type': influence_type,
            'global_params': {param: sim_params.get(param) for param in affected_params if param in sim_params}
        }
        
        # Add computed contexts based on influence type
        if influence_type == 'behavioral':
            context.update(self._compute_behavioral_context(sim_params, decision_name))
        elif influence_type == 'strategic':
            context.update(self._compute_strategic_context(sim_params, decision_name))
        elif influence_type == 'constraint':
            context.update(self._compute_constraint_context(sim_params, decision_name))
        elif influence_type == 'choice':
            context.update(self._compute_choice_context(sim_params, decision_name))
        
        return context
    
    def _compute_behavioral_context(self, sim_params: Dict[str, Any], decision_name: str) -> Dict[str, Any]:
        """Compute behavioral factors from global parameters"""
        context = {}
        
        if decision_name == 'donation_default':
            # Income distribution effects on prosocial behavior
            income_dist = sim_params.get('income_distribution', 'lognormal')
            if income_dist == 'pareto':
                # Pareto distribution = high inequality = potentially higher donations from wealthy
                context['inequality_factor'] = 1.2
            elif income_dist == 'weibull':
                # Weibull = more equal distribution = standard donations
                context['inequality_factor'] = 1.0
            else:  # lognormal
                # Lognormal = moderate inequality = baseline
                context['inequality_factor'] = 1.0
            
            # Market competition effects
            num_vendors = sim_params.get('num_vendors', 5)
            context['competition_factor'] = 1.0 + (num_vendors - 5) * 0.02  # More vendors = slightly higher donations
            
            # Platform trust effects
            platform_markup = sim_params.get('platform_markup', 0.1)
            context['trust_factor'] = 1.0 - platform_markup * 0.5  # Higher markup = lower trust = lower donations
            
            # Income level effects
            income_avg = sim_params.get('income_avg', 5000)
            income_min = sim_params.get('income_min', 1000)
            # Normalize average income to get prosperity factor
            context['prosperity_factor'] = min(1.5, income_avg / income_min / 5.0)
            
            # Multi-period effects
            periods = sim_params.get('periods', 1)
            if periods > 1:
                # Multi-period game = more strategic = slightly lower initial donations
                context['temporal_factor'] = 0.95
            else:
                context['temporal_factor'] = 1.0
                
        return context
    
    def _compute_strategic_context(self, sim_params: Dict[str, Any], decision_name: str) -> Dict[str, Any]:
        """Compute strategic factors from global parameters"""
        context = {}
        
        if decision_name in ['disclose_income', 'disclose_documents']:
            # Category competition effects
            num_discount = sim_params.get('num_discount_categories', 3)
            num_fixed = sim_params.get('num_fixed_categories', 5)
            
            # More categories = more incentive to disclose
            context['category_competition'] = (num_discount + num_fixed) / 8.0
            
            # Income distribution affects disclosure strategy
            income_dist = sim_params.get('income_distribution', 'lognormal')
            if income_dist == 'pareto':
                # High inequality = more strategic about disclosure
                context['disclosure_incentive'] = 0.8
            else:
                context['disclosure_incentive'] = 1.0
                
        elif decision_name == 'purchase_vs_bid':
            # Bidding availability
            bid_percentage = sim_params.get('bidding_percentage', 0.5)
            context['auction_availability'] = bid_percentage
            
            # Price flexibility
            price_range = sim_params.get('price_range', 0.25)
            context['price_flexibility'] = price_range
            
            # Income effects on risk tolerance
            income_avg = sim_params.get('income_avg', 5000)
            context['risk_tolerance'] = min(1.0, income_avg / 5000)
            
        return context
    
    def _compute_constraint_context(self, sim_params: Dict[str, Any], decision_name: str) -> Dict[str, Any]:
        """Compute constraint-based factors from global parameters"""
        context = {}
        
        if decision_name == 'consumption_quantity':
            # Budget constraint
            income_avg = sim_params.get('income_avg', 5000)
            market_price = sim_params.get('market_price', 10.0)
            context['budget_ratio'] = income_avg / (market_price * 100)  # Income vs 100 products
            
            # Time constraint
            periods = sim_params.get('periods', 1)
            duration_hours = sim_params.get('duration_hours', 1.0)
            context['time_pressure'] = 1.0 / (periods * duration_hours)
            
            # Availability constraint
            products_per_vendor = sim_params.get('products_per_vendor', 100)
            num_vendors = sim_params.get('num_vendors', 5)
            context['availability_ratio'] = (products_per_vendor * num_vendors) / 500  # Normalized to baseline
            
        return context
    
    def _compute_choice_context(self, sim_params: Dict[str, Any], decision_name: str) -> Dict[str, Any]:
        """Compute choice-related factors from global parameters"""
        context = {}
        
        if decision_name == 'vendor_selection':
            # Choice set size
            num_vendors = sim_params.get('num_vendors', 5)
            context['choice_complexity'] = np.log(num_vendors + 1)  # Logarithmic complexity
            
            # Price variation
            price_min = sim_params.get('vendor_price_min', 8.0)
            price_max = sim_params.get('vendor_price_max', 12.0)
            market_price = sim_params.get('market_price', 10.0)
            context['price_variance'] = (price_max - price_min) / market_price
            
            # Price source effects
            price_source = sim_params.get('vendor_price_source', 'random')
            context['price_predictability'] = 0.5 if price_source == 'random' else 1.0
            
        return context
    
    def validate_parameters(self, simulation_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate global parameters and return any warnings or errors
        
        Args:
            simulation_config: Global simulation configuration
            
        Returns:
            Dictionary with 'valid' boolean and 'issues' list
        """
        issues = []
        sim_params = simulation_config.get('simulation', {})
        
        # Validate price relationships
        price_min = sim_params.get('vendor_price_min', 0)
        price_max = sim_params.get('vendor_price_max', 0)
        market_price = sim_params.get('market_price', 0)
        
        if price_min > market_price:
            issues.append("vendor_price_min should not exceed market_price")
        if price_max < market_price:
            issues.append("vendor_price_max should not be below market_price")
        if price_min >= price_max:
            issues.append("vendor_price_min should be less than vendor_price_max")
            
        # Validate income relationships
        income_min = sim_params.get('income_min', 0)
        income_max = sim_params.get('income_max', 0)
        income_avg = sim_params.get('income_avg', 0)
        
        if income_min >= income_max:
            issues.append("income_min should be less than income_max")
        if income_avg < income_min or income_avg > income_max:
            issues.append("income_avg should be between income_min and income_max")
            
        # Validate other parameters
        if sim_params.get('price_grid', 11) % 2 == 0:
            issues.append("price_grid must be an odd number")
            
        if sim_params.get('bidding_percentage', 0.5) < 0 or sim_params.get('bidding_percentage', 0.5) > 1:
            issues.append("bidding_percentage must be between 0 and 1")
            
        return {
            'valid': len(issues) == 0,
            'issues': issues
        }
