"""
Test script to verify parameter persistence issue

Run this to check if parameters are being stored correctly in session state
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[0]))

# Simulate session state behavior
class SessionState:
    def __init__(self):
        self.data = {}
    
    def __setattr__(self, key, value):
        if key == 'data':
            super().__setattr__(key, value)
        else:
            self.data[key] = value
    
    def __getattr__(self, key):
        if key == 'data':
            return super().__getattribute__(key)
        return self.data.get(key)
    
    def __contains__(self, key):
        return key in self.data
    
    def get(self, key, default=None):
        return self.data.get(key, default)


def test_widget_pattern():
    """Test the current widget pattern"""
    print("=" * 60)
    print("Testing Current Widget Pattern")
    print("=" * 60)
    
    session_state = SessionState()
    
    # Simulate SimulationParameters
    class SimParams:
        def __init__(self):
            self.periods = 1  # Default
    
    session_state.sim_params = SimParams()
    
    print("\n1. Initial state:")
    print(f"   sim_params.periods = {session_state.sim_params.periods}")
    
    # Simulate first page render (user changes value to 5)
    print("\n2. User changes periods to 5 (first render):")
    # Widget creates key
    session_state.periods_input = 5
    # on_change callback
    session_state.sim_params.periods = session_state.periods_input
    print(f"   periods_input = {session_state.periods_input}")
    print(f"   sim_params.periods = {session_state.sim_params.periods}")
    
    # Simulate navigation (script rerun)
    print("\n3. Navigate to Page 2 and back (script rerun):")
    print("   Widget reinitializes with: value=st.session_state.sim_params.periods")
    print(f"   Widget shows: {session_state.sim_params.periods}")
    print(f"   periods_input still has: {session_state.periods_input}")
    
    # BUT if sim_params got reset...
    print("\n4. If sim_params.periods somehow got reset to default:")
    session_state.sim_params.periods = 1  # RESET
    print(f"   sim_params.periods = {session_state.sim_params.periods}")
    print(f"   periods_input still has: {session_state.periods_input}")
    print(f"   But widget value= reads from: sim_params.periods = {session_state.sim_params.periods}")
    print("   ❌ USER SEES: 1 (lost their change of 5!)")
    
    print("\n" + "=" * 60)
    print("Solution: Widget should read from its own key")
    print("=" * 60)
    
    # Reset for solution test
    session_state = SessionState()
    session_state.sim_params = SimParams()
    
    print("\n1. Initial state:")
    print(f"   sim_params.periods = {session_state.sim_params.periods}")
    
    # First render with fix
    print("\n2. First render with FIXED pattern:")
    if "periods_input" not in session_state:
        session_state.periods_input = session_state.sim_params.periods
    print(f"   Initialized periods_input from sim_params: {session_state.periods_input}")
    
    print("\n3. User changes periods to 5:")
    session_state.periods_input = 5
    session_state.sim_params.periods = session_state.periods_input
    print(f"   periods_input = {session_state.periods_input}")
    print(f"   sim_params.periods = {session_state.sim_params.periods}")
    
    print("\n4. Navigate to Page 2 and back (script rerun):")
    # Check if key exists (it does)
    if "periods_input" not in session_state:
        print("   Would initialize from sim_params")
    else:
        print("   Key exists, skipping initialization")
    print(f"   Widget reads from: periods_input = {session_state.periods_input}")
    print("   ✅ USER SEES: 5 (preserved!)")
    
    print("\n5. Even if sim_params.periods got reset:")
    session_state.sim_params.periods = 1  # RESET
    print(f"   sim_params.periods = {session_state.sim_params.periods}")
    print(f"   Widget still reads from: periods_input = {session_state.periods_input}")
    print("   ✅ USER STILL SEES: 5 (widget key protects the value!)")


if __name__ == "__main__":
    test_widget_pattern()

