# ⚠️ Income Category Assignment - Temporary Implementation

## Current Status

**The income category assignments currently shown in the "Quantity by Income Category" visualization are NOT representative of the final implementation.**

---

## What You're Seeing Now

The current simulation uses a **temporary category assignment logic** that:

1. Divides the theoretical income range into equal intervals
2. Assigns agents to categories based on their generated income
3. Results in **some categories being empty** (e.g., categories 8-10 when NFIC=10)

**This is temporary placeholder logic** and does not reflect the intended production behavior.

---

## What Will Change

### Future Implementation

In the production version, **income category assignments will be read from an external algorithm** that:

- Uses a more sophisticated categorization method
- Properly handles the relationship between:
  - Assigned Allowance Level (1-5 from copula)
  - Generated income values
  - Income categories (1 to NFIC)
- Ensures appropriate population of all categories
- Aligns with experimental design requirements

### Integration Plan

The simulation will be **connected to external code** that:
1. Calculates income categories using the proper algorithm
2. Provides category assignments as input to the simulation
3. Ensures logical consistency across all income-related variables

---

## Impact on Current Results

### What This Means for Your Analysis

❌ **Do NOT use** the current "Quantity by Income Category" data for:
- Final research conclusions
- Publication figures
- Policy recommendations
- Comparative analysis across income levels

✅ **You CAN use** other simulation outputs:
- Customer type distributions (Discount/Fixed/Regular)
- Purchase patterns by customer type
- Vendor selection behavior
- Bidding behavior
- Donation rates

### Known Issues with Current Categories

1. **Empty Categories**: Some high-numbered categories have no agents
   - This occurs because the temporary logic uses equal-width intervals
   - Agents generated from 5 quintile buckets don't reach the highest intervals

2. **Inconsistent Population**: Categories have highly uneven agent counts
   - Category 1: ~310 agents
   - Category 2: ~440 agents
   - Categories 8-10: 0 agents

3. **Misalignment**: The category boundaries don't align with:
   - The 5 Assigned Allowance Levels
   - The percentile buckets used for income generation
   - Meaningful economic thresholds

---

## Timeline

**Current State**: Temporary placeholder logic (not representative)

**Next Step**: Integration with external category assignment algorithm

**Expected**: Category assignments will be provided by external code before production use

---

## Questions?

If you have questions about:
- When the external algorithm will be integrated
- How the new categorization will work
- Whether specific analysis is affected

Please consult with the development team before using income category data.

---

**Last Updated**: November 12, 2025  
**Status**: ⚠️ TEMPORARY - NOT REPRESENTATIVE

