# Reference documents

The platform documentation is the source of truth for the model and for every exported
number. When the code and a document disagree, the document wins; the few deliberate
deviations are listed in [`../README.md`](../README.md) §9.
Newer versions of these files replace the copies here; keep the date in the file name.

| File | Date | What it is |
|---|---|---|
| `COOPECON AI Agent Documentation 2026-10-07.docx` | 2026-10-07 | The rules of this simulation: Page 1 parameters, the thirteen decisions with their equations and constants, the σ model, and the exact columns of every Excel export. **Read this first.** |
| `Cooperative Economy Documentation 2026-06-19.docx` | 2026-06-19 | The COOPECON platform itself: pricing (Pc, Ppn, Pmb, the price grid), customer types, transaction processing, rejected-transaction options, the optimiser. |
| `Output Documentation File 2026-06-18.docx` | 2026-06-18 | The platform's own output report format, period by period. |

Where the agent documentation settles something the code used to leave open, the
place is noted in the code with the document name and section, for example the
Customer Price rule in `app/reports/transaction_level.py` and the vendor score in
`app/reports/agent_level.py`.
