# Where to Find the Donation Rate Excel Download Button

## 📥 Updated: Button Now Always Visible!

The Excel download button has been updated and will now **always appear** if you have purchase requests in your simulation.

---

## 🎯 How to Access

### **Step 1: Run Simulation**

On **Page 2**, select these decisions:
- ✅ `purchasing_quantity` (REQUIRED - creates purchase requests)
- ✅ `final_donation_rate` (for donation rate data)
- ✅ `donation_default` (OPTIONAL - for complex rates)

Click **"🚀 Run Simulation"**

---

### **Step 2: Go to Results Page**

Click **"📊 View Results"** or navigate to the Results tab

---

### **Step 3: Find Final Donation Rate Section**

Scroll down to the **"Final Donation Rate"** section

You'll see one of two scenarios:

#### **Scenario A: With Donation Configuration** (when `donation_default` is selected)
```
📊 Final Donation Rate (Custom Parameters)

📊 Using Distribution from Selected Donation Configuration
✅ The final_donation_rate values in your export match the donation_default distribution

[Donation Rate Statistics: Mean, Median, Std Dev]
[Histogram of donation rates]

───────────────────────────────────────────────

💾 Transaction-Level Export
Download detailed purchase request data with donation rates

[Metrics: Total Requests | Total Agents | Periods | Avg Donation Rate]

📥 Download Transaction-Level Excel  ← CLICK HERE!
```

#### **Scenario B: Without Donation Configuration** (simple default values)
```
📊 Final Donation Rate (Custom Parameters)

💡 No donation configuration selected - Using simple rate configuration

[Simple metrics]

───────────────────────────────────────────────

💾 Transaction-Level Export
Download detailed purchase request data with donation rates

[Metrics: Total Requests | Total Agents | Periods | Avg Donation Rate]

📥 Download Transaction-Level Excel  ← CLICK HERE!
```

---

## ✅ What You'll Get

### **Excel File Structure:**

**File Name:** `donation_transactions_YYYYMMDD_HHMMSS.xlsx`

**Sheets:**
1. **Total** - All purchase requests across all periods
2. **Period 1** - Requests from period 1 only
3. **Period 2** - Requests from period 2 only
4. ... one sheet per period

**Columns in Each Sheet:**
| Column | Description | Source |
|--------|-------------|--------|
| Agent ID | Which agent | Agent-level |
| Assigned Allowance Level | Income level | Agent-level |
| Group_experiment | Experimental group | Agent-level |
| Customer Type | Regular/Fixed/Discount | Request-level |
| Income Category | Category 1-N | Agent-level |
| Purchase Request Type | PN/Bid/Fixed/Discount | Request-level |
| Date/Time of Purchase Request | When request occurred | Request-level |
| Period | Period number | Request-level |
| Customer Price | Price for this request | Request-level |
| Transaction Completed | 0/1 or N/A | Request-level |
| Default Donation Rate | Agent's baseline | Agent-level |
| **Final Donation Rate** | **Donation rate for THIS request** | **Request-level** ✅ |
| Donation Paid | Price × Rate | Calculated |
| Total Paid by Customer | Price + Donation | Calculated |

---

## 🔍 Example Excel Data

```excel
| Agent ID | Period | Customer Price | Final Donation Rate | Donation Paid | Total Paid |
|----------|--------|----------------|---------------------|---------------|------------|
| 1        | 1      | 110.00         | 0.173               | 19.03         | 129.03     |
| 1        | 1      | 95.50          | 0.173               | 16.52         | 112.02     |
| 1        | 1      | 110.00         | 0.173               | 19.03         | 129.03     |
| 2        | 1      | 110.00         | 0.100               | 11.00         | 121.00     |
| 2        | 1      | 70.00          | 0.100               | 7.00          | 77.00      |
```

---

## ❓ Troubleshooting

### **Q: I don't see the download button!**

**A:** Make sure you selected `purchasing_quantity` decision on Page 2. The button only appears if there are purchase requests in your simulation.

### **Q: I see "No purchase request data found"**

**A:** The simulation didn't create any purchase requests. This happens if:
- `purchasing_quantity` decision wasn't selected
- All agents got quantity = 0 (randomly)
- The simulation failed

**Solution:** Re-run with `purchasing_quantity` selected.

### **Q: The button is there but grayed out**

**A:** This might be a Streamlit session issue. Try:
1. Refresh the page (Ctrl+R or Cmd+R)
2. Re-run the simulation
3. Clear cache (Settings → Clear Cache)

### **Q: Download button appears twice?**

**A:** There might be a caching issue. Refresh the page. If it persists, this is now fixed in the latest version.

---

## 📋 Quick Checklist

Before looking for the download button:

- [ ] Simulation has been run
- [ ] `purchasing_quantity` decision was selected on Page 2
- [ ] You're on the Results page
- [ ] You've scrolled to "Final Donation Rate" section
- [ ] The section shows metrics (Total Requests, Total Agents, etc.)

If all checked ✅ → Download button should be visible below the metrics!

---

## 🎯 Summary

**Location:** Results Page → Final Donation Rate Section → Transaction-Level Export  
**Button Text:** 📥 Download Transaction-Level Excel  
**Always Visible:** Yes (as long as purchase requests exist)  
**File Format:** .xlsx with multiple sheets (Total + per Period)  

**The download button is now always visible when you have purchase request data!** 🎉

