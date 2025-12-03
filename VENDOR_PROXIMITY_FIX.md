# Vendor Proximity Score Fix

## Issue
Vendor proximity scores were artificially increasing linearly with Vendor ID (e.g., Vendor 1 ~20, Vendor 6 ~80). This was caused by assigning sorted "location means" to vendors in sequential order.

## Fix Implemented
**Date:** November 28, 2025
**File:** `src/vendor_attribute_generator.py`

Implemented **Consistent "Random" Locations** (Option 1):
- Added a deterministic shuffle to the `vendor_means` array before assignment.
- Used `random.Random(42 + num_vendors)` to seed the shuffle.

## Result
- **Distribution Preserved:** The simulation still contains the same mix of "Urban" (high proximity), "Suburban" (medium), and "Rural" (low) vendors.
- **Correlation Broken:** High Vendor IDs no longer correlate with better location scores.
- **Consistency Maintained:** All agents in the simulation still agree on which specific vendor is "Urban" vs "Rural" (e.g., Vendor 3 might always be the urban one for a given run).

This change is purely cosmetic/distributional and does not affect the underlying mechanics of the simulation, but it makes the results look more realistic and less like a software artifact.



