# Consumption Limit Issue: Limits Apply to COMPLETED TRANSACTIONS, Not Requests

**Date:** November 28, 2025  
**Status:** 🔴 DESIGN ISSUE - Needs Architecture Update  
**Issue Number:** #10

---

## 🎯 The Problem

### Current Implementation (INCORRECT)
```python
# Line 401-406 in src/decisions/purchasing_quantity.py
# STEP 5: Generate total quantity for the term
# Random integer in [0, purchasing_limit] inclusive
if purchasing_limit > 0:
    total_quantity = int(rng.integers(0, purchasing_limit + 1))
else:
    total_quantity = 0
```

**What this does:**
- Limits the NUMBER OF PURCHASE REQUESTS to the consumption limit
- If limit = 50, agent generates 0-50 purchase requests
- Assumes all requests will complete

### Correct Specification (from Professor)

**The consumption limit applies to COMPLETED TRANSACTIONS, not to PURCHASE REQUESTS.**

**Example:**
```
Agent's consumption limit: 50 items per term

Scenario 1 (Currently works):
- Agent makes 30 purchase requests
- All 30 complete
- Total completed: 30 ✅ Within limit (30 ≤ 50)

Scenario 2 (Should be allowed but currently blocked):
- Agent makes 100 purchase requests
- Only 50% complete (50 complete, 50 rejected)
- Total completed: 50 ✅ Within limit (50 ≤ 50)
- This is VALID but current code would never generate 100 requests

Scenario 3 (Should be allowed but currently blocked):
- Agent makes 80 purchase requests
- Only 60% complete (48 complete, 32 rejected)
- Total completed: 48 ✅ Within limit (48 ≤ 50)
- This is VALID but current code would never generate 80 requests
```

---

## 📋 Why This Matters

### In Reality:
1. Agents don't know in advance which requests will complete
2. Agents might make MORE requests than their limit, anticipating rejections
3. The limit is an ex-post constraint on COMPLETED purchases, not an ex-ante constraint on REQUESTS

### Current System:
- ❌ Treats requests and completed transactions as the same
- ❌ Limits requests instead of completions
- ❌ No simulation of transaction completion/rejection

---

## 🔧 What Needs to Change

### Phase 1: Immediate (When Decision 10 is NOT simulated)

**Keep current behavior as default:**
- Generate requests up to limit
- Assume 100% completion rate
- This is the "default behavior" when we don't simulate transaction outcomes

**But add clarification in code comments and UI:**
```python
# STEP 5: Generate total quantity for the term
# NOTE: This is DEFAULT BEHAVIOR when Decision 10 (rejected_transaction_option) is not simulated
# In reality, the limit applies to COMPLETED transactions, not requests
# When fully simulated, agents may make MORE requests than the limit,
# knowing that some will be rejected
if purchasing_limit > 0:
    total_quantity = int(rng.integers(0, purchasing_limit + 1))
else:
    total_quantity = 0
```

### Phase 2: When Decision 10 IS Simulated (Future)

**New Architecture:**
```python
def purchasing_quantity_with_rejection_simulation(agent_state, params, rng, simulation_config):
    """
    When simulating transaction outcomes:
    1. Agent generates purchase requests (can exceed limit)
    2. System simulates which requests complete vs reject
    3. Enforce: count(completed) <= consumption_limit
    """
    
    # Agent's desired quantity (ignoring limit for now)
    # This could be based on budget, utility, or other factors
    desired_quantity = _calculate_desired_quantity(agent_state, rng, simulation_config)
    
    # Generate all desired purchase requests
    purchase_requests = _generate_requests(desired_quantity, ...)
    
    # Simulate which requests complete vs reject
    # This happens in Decision 10 or a separate transaction processing step
    completed_requests, rejected_requests = _simulate_transaction_outcomes(
        purchase_requests, 
        consumption_limit,
        simulation_config
    )
    
    # Ensure completed count doesn't exceed limit
    assert len(completed_requests) <= consumption_limit
    
    return {
        "purchasing_quantity": desired_quantity,
        "purchase_requests": purchase_requests,
        "completed_transactions": completed_requests,
        "rejected_transactions": rejected_requests
    }
```

---

## 📊 Impact on Current System

### What Changes Now:
1. **Documentation**: Update code comments to clarify this is default behavior
2. **UI Text**: Update frontend to explain this limitation
3. **Architecture**: Acknowledge this is simplified for default behavior

### What Changes Later (When Decision 10 is Simulated):
1. **Request Generation**: Remove limit from request generation
2. **Transaction Simulation**: Add new step to simulate completions/rejections
3. **Limit Enforcement**: Move limit check to completed transactions
4. **UI Display**: Show both requests AND completion rates

---

## 🔍 Current State in Codebase

### File: `src/decisions/purchasing_quantity.py`

**Lines 401-406:**
```python
# STEP 5: Generate total quantity for the term
# Random integer in [0, purchasing_limit] inclusive
if purchasing_limit > 0:
    total_quantity = int(rng.integers(0, purchasing_limit + 1))
else:
    total_quantity = 0
```

**This is where the incorrect assumption is made.**

### File: `app/pages/results/visualizations/purchasing_viz.py`

**Lines 321-326:** Shows note that completion data is not available
```python
st.info(
    "ℹ️ **Note:** Completed transaction information is not currently available from the algorithm. "
    "Currently showing all purchase requests as completed (100% completion rate). "
    "Once the algorithm provides actual completion/rejection data, it will be imported and displayed here."
)
```

**This note acknowledges the limitation but doesn't explain the consumption limit issue.**

---

## ✅ Recommended Actions

### Immediate (Documentation):
1. ✅ Add comments to `purchasing_quantity.py` explaining this is default behavior
2. ✅ Update UI text to clarify limits currently apply to requests (not transactions)
3. ✅ Create this documentation file

### Future (When Ready to Simulate Decision 10):
1. ⏳ Design transaction outcome simulation
2. ⏳ Decouple request generation from limit enforcement
3. ⏳ Add completion/rejection tracking
4. ⏳ Update UI to show both requests and completion rates

---

## 🎓 Professor's Clarification (Exact Quote)

> "The limit applies to completed transactions, not to purchase requests. So what matters in terms of my consumption limits is that I do not make if the limit is 50, then I do not make more than 50 purchases, but it is possible that I would make 100 purchases and only 50% of them will be completed. So this is OK with the purchase limit. So at the moment we are generating requests up to the purchase limit."

---

## 📝 Summary

- **Current System**: Limits REQUESTS to the consumption limit (simplified default)
- **Correct System**: Should limit COMPLETED TRANSACTIONS, not requests
- **Why Different**: We don't currently simulate which requests complete vs reject
- **What to Do**: 
  - Keep current behavior as default (when Decision 10 not simulated)
  - Add clear documentation explaining this simplification
  - Plan for future enhancement when transaction outcomes are simulated

