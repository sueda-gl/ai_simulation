# src/parameter_applicability.py
import yaml
from pathlib import Path
from typing import Dict, List, Set, Optional
from dataclasses import dataclass

@dataclass
class ParameterInfo:
    """Information about a parameter's applicability for a specific decision"""
    name: str
    is_applicable: bool
    category: str
    not_applicable_reason: str = ""

class ParameterApplicabilityManager:
    """Manages parameter applicability for different decisions"""
    
    def __init__(self, config_path: str = "config/parameter_applicability.yaml"):
        self.config_path = Path(config_path)
        self.config = self._load_config()
        self.parameter_categories = self.config.get('parameter_categories', {})
        self.all_parameters = self._extract_all_parameters()
    
    def _load_config(self) -> dict:
        """Load the parameter applicability configuration"""
        try:
            with open(self.config_path, 'r') as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            print(f"Warning: Parameter applicability config not found at {self.config_path}")
            return {"decisions": {}, "parameter_categories": {}}
        except yaml.YAMLError as e:
            print(f"Error parsing parameter applicability config: {e}")
            return {"decisions": {}, "parameter_categories": {}}
    
    def _extract_all_parameters(self) -> Set[str]:
        """Extract all unique parameters from the category definitions"""
        all_params = set()
        for category_params in self.parameter_categories.values():
            all_params.update(category_params)
        return all_params
    
    def get_applicable_parameters(self, decision: str) -> List[str]:
        """Get list of applicable parameters for a decision"""
        decision_config = self.config.get('decisions', {}).get(decision, {})
        return decision_config.get('applicable_parameters', [])
    
    def get_not_applicable_parameters(self, decision: str) -> List[str]:
        """Get list of parameters that are NOT applicable for a decision"""
        applicable = set(self.get_applicable_parameters(decision))
        return list(self.all_parameters - applicable)
    
    def is_parameter_applicable(self, decision: str, parameter: str) -> bool:
        """Check if a parameter is applicable for a specific decision"""
        return parameter in self.get_applicable_parameters(decision)
    
    def get_parameter_info(self, decision: str, parameter: str) -> ParameterInfo:
        """Get detailed information about a parameter's applicability"""
        is_applicable = self.is_parameter_applicable(decision, parameter)
        category = self._get_parameter_category(parameter)
        not_applicable_reason = ""
        
        if not is_applicable:
            decision_config = self.config.get('decisions', {}).get(decision, {})
            not_applicable_reason = decision_config.get(
                'not_applicable_reason', 
                f"Parameter not applicable for {decision}"
            )
        
        return ParameterInfo(
            name=parameter,
            is_applicable=is_applicable,
            category=category,
            not_applicable_reason=not_applicable_reason
        )
    
    def _get_parameter_category(self, parameter: str) -> str:
        """Get the category of a parameter"""
        for category, params in self.parameter_categories.items():
            if parameter in params:
                return category.replace('_', ' ').title()
        return "Uncategorized"
    
    def get_parameters_by_category(self, decision: str) -> Dict[str, Dict[str, List[ParameterInfo]]]:
        """Get parameters organized by category with applicability info"""
        result = {}
        
        for category, params in self.parameter_categories.items():
            category_name = category.replace('_', ' ').title()
            result[category_name] = {
                'applicable': [],
                'not_applicable': []
            }
            
            for param in params:
                param_info = self.get_parameter_info(decision, param)
                if param_info.is_applicable:
                    result[category_name]['applicable'].append(param_info)
                else:
                    result[category_name]['not_applicable'].append(param_info)
        
        return result
    
    def get_decision_summary(self, decision: str) -> Dict[str, any]:
        """Get a summary of parameter applicability for a decision"""
        applicable = self.get_applicable_parameters(decision)
        not_applicable = self.get_not_applicable_parameters(decision)
        decision_config = self.config.get('decisions', {}).get(decision, {})
        
        return {
            'decision': decision,
            'total_parameters': len(self.all_parameters),
            'applicable_count': len(applicable),
            'not_applicable_count': len(not_applicable),
            'applicable_parameters': applicable,
            'not_applicable_parameters': not_applicable,
            'reason': decision_config.get('not_applicable_reason', ''),
            'applicability_ratio': len(applicable) / len(self.all_parameters) if self.all_parameters else 0
        }
    
    def get_all_decisions_summary(self) -> Dict[str, Dict]:
        """Get applicability summary for all decisions"""
        return {
            decision: self.get_decision_summary(decision)
            for decision in self.config.get('decisions', {}).keys()
        }
    
    def update_decision_applicability(self, decision: str, applicable_parameters: List[str], 
                                    not_applicable_reason: str = ""):
        """Update the applicability configuration for a decision"""
        if 'decisions' not in self.config:
            self.config['decisions'] = {}
        
        self.config['decisions'][decision] = {
            'applicable_parameters': applicable_parameters,
            'not_applicable_reason': not_applicable_reason
        }
        
        # Save back to file
        self._save_config()
    
    def _save_config(self):
        """Save the configuration back to the YAML file"""
        try:
            with open(self.config_path, 'w') as f:
                yaml.dump(self.config, f, default_flow_style=False, sort_keys=False)
        except Exception as e:
            print(f"Error saving parameter applicability config: {e}")

# Global instance for use throughout the application
param_manager = ParameterApplicabilityManager()
