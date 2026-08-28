# Decision 4 — Numerical Error Review Guide (verified against revision 260826-2)

Usage: each entry's **search string** is a literal Ctrl+F string for Word — paste it as-is to jump to the error. Status verified against the accepted-text state of `Decision 4 - Rejected Transaction Defaults 260826-2.docx` (tracked deletions treated as fixed). Fixed items are listed at the bottom.

## Propagating errors (change final coefficients) — REMAINING

| id | search | section | printed | correct | impact |
|----|--------|---------|---------|---------|--------|
| L2 | `0.023723` | §2 Loyalty | Ho&Wong SE(ESTLtotal) = 0.023723 | delta-method ≈0.140–0.170 (full first-order delta 0.16985) | ≈0.024 looks like 0.158737×0.151515 — two SEs multiplied instead of a delta-method combination; still gives H&W weight 1776.891 (should be ≈35) inside the NEW recomputed Extraversion pool (βcombined 0.036745), so the rebuilt loyalty equation still inherits it. |
| F8 | `0.378 × 0.3621291` | §5 Flexibility | Daban N β = 0.378 in the combination line | 0.376 (the study's own tabled β) | β_N,combined should be −0.3598070, not −0.3605313; final IVW N coefficient −0.0537819, not the printed −0.05389019. |
| F9 | `0.00911805288` | §5 Flexibility | A→Flex H&S Var(β̄) = 0.00911805288 → SE 0.0954885 | 6.1304861674/5193 = 0.0011805288 → SE 0.0343588 | Digit '9' inserted; correct z 5.42, p≈6e-8 — A is highly significant under H&S, undercutting the doc's stated rationale for preferring IVW. |
| F10 | `0.0363350905` | §5 Flexibility | C combined SE = 0.0363350905 (Daban's 0.297² used twice in place of Latzman's 0.1264²) | 0.0289851 | The revision fixed the table's 1/SE² cell to 1190.283722 (= 1/0.0289851², correct) but left the SE at 0.0363351 AND still uses 757.43875 in the cross-trait sum (4708.16127478) — so w_C and ALL five final IVW weights/coefficients are still wrong; with the fix Σ = 5141.006 and w_C = 0.23153. |
| N1 | `5.2667576` | §3 WTP (NEW in 260826-2) | Rel_weight_A = 251/4765 = 5.2667576% | 251/4765 = 5.267576% | Digit slip inside the user's 4765 correction; propagates into the final A coefficient: printed −0.0123242128 (= 0.052667576×0.234 exactly) should be −0.0123261280, and into every WTP code line reusing it. Magnitude tiny (5th decimal); the re-run categorical regression reproduces from the .dta with the printed value. |

## Confined errors (dropped traits / no downstream effect) — REMAINING

| id | search | section | printed | correct | impact |
|----|--------|---------|---------|---------|--------|
| W2r | `0.034438` | §3 WTP summary table | Openness SE 0.034438 | 0.0319155 | The derivation line was corrected (√0.0010186 = 0.0319155 ✓) but the summary-table cell still shows the old wrong SE; trait dropped either way. |
| W3 | `0.272230` | §3 WTP | C Udo-Imeh SE 0.272230 (stated z=1.645) | 0.2122 | 0.272230 implies one-tailed z=1.282, not the stated 1.645; trait still dropped. |
| R1 | `0.332208167` | §4 RT | σ_overall = (5.979747/18)*mean(stdactions) = 5.979747/18 = 0.332208167 | ×1.1863376 = 0.394111 | The formula now explicitly writes ×mean(stdactions) but the computation still omits it; the loyalty (0.392584948) and flexibility (0.428012086) σ in this revision DO multiply — RT is the only inconsistent one. |

## Typos & presentation — REMAINING

| id | search | section | issue |
|----|--------|---------|-------|
| L7 | `0.347` | §2 summary table | Table shows L&D mediator ×0.347; all calculations use ×0.366. |
| L8 | `(0.93+0.63)` | §2 summary table | H&W multipliers shown vs the 0.62/0.34/0.66 actually used. |
| W6 | `z = 4.2` | §3 | Misplaced combined z in study row; the SE derives from z=3.291. |
| W7 | `0.064687` | §3 | N combined-SE input vs the table's 0.064722; also 101.61 vs 101.601 (477×0.213). Trivial. |
| W9 | `(-.691843 +)` | §3 | Inline "Therefore:" pseudo-Stata line regenerated with the new numbers but still garbled (unbalanced parentheses); the operative cond() code below is fine apart from W8. |
| R2 | `(-.0068307 +)` | §4 | Same garbled inline "Therefore:" line in RT. |
| F11 | `0.0293893299` | §5 | C H&S deviations: 0.21−0.2396893299 = −0.0296893299 (printed −0.0293893299); next line −0.1093893299 should be −0.1096893299 → SE 0.0496803 vs printed 0.04963272 (conclusion unchanged). |
| F12 | `1302 + 1731 + 1731` | §5 | ΣW listing has "+1731" five times (= 9957); the total 8226 = 1302+4×1731 actually used is right — the listing is wrong. |
| F13 | `0.000922639` | §5 O→Flex | Derivation still displays Extraversion's numbers: "√0.000922639" (the four O components correctly sum to 0.0011290623) and "z = 0.09771 / 0.030375 = 5.080758; p = 0.0013" (0.09771/0.030375 = 3.22; the z 5.080758 = O's own 0.1707212/0.03360152, and p should be ≈3.7e-7, not 0.0013). Tabled O results are right. |
| F14 | `p > 0.0001` | §5 | Should be "<". |
| N2 | `7500.29394` | §2 (NEW in 260826-2) | Text line "1/ SE(βcombined)² = 7500.29394" contradicts the table's (correct) 7500.28394 — one-digit slip in the corrected Agreeableness block. |

## Code bugs in Stata listings — REMAINING

| id | search | section | bug | fix |
|----|--------|---------|-----|-----|
| W8 | `0.0123242128 * agreeable` | §3 | weighted_WTP_categorical cond(): the level-1 (base-allowance) branch uses raw `agreeable` instead of the z-score — shifts level-1 scores by ≈−0.06 to −0.03. | `z_agreeable` |

## Specification questions (professor must rule) — REMAINING

| id | search | section | question |
|----|--------|---------|----------|
| W10 | `.5439353` | §3 | The constructed weighted_WTP_categorical applies the personality part at coefficient 1.0, silently discarding the fitted slope 0.5439353 (regression verified bit-for-bit from the .dta) — harmless in RT (slope 0.9991) but materially re-weights personality vs income in WTP. Intentional? |
| F16 | `49.53` | §5 | Daban's Extraversion effect (β=−0.044, p=0.308) is included with the LARGEST weight (49.53%) although the stated rule — and the section's own sentence — say non-significant effects are excluded. Excluding it gives β_E = 0.2368 instead of 0.09771. |

## Verified fixed in revision 260826-2

All of the following were spot-checked by recomputation; the corrections are arithmetically right except where noted.

- **L1** — L&D Extraversion weight now 13091.0614 = 1/0.0087400² ✓ (SE cell now 0.00874); whole E pool recomputed (Σweights 15129.45877, βcombined 0.036745 — internally consistent, but still inherits L2).
- **L3** — Agreeableness combined SE now 0.0115456, z 3.98944535, p<0.0001, Σweights 7500.28394 ✓ (new one-digit slip N2 in the text line below the table).
- **L4** — Azzahra Openness weight now 114.878 ✓.
- **L5** — superseded: the loyalty σ block was rewritten as (5.95659/18)×mean(stdactions) = 0.392584948 ✓ (now includes the multiplier); the old €32-row table value is gone.
- **L6** — superseded: old Σweights 2152.812872 row no longer exists (E pool recomputed).
- **W1** — H&S sum now 4765 throughout, rel. weights 21.259182 / 5.2667576 (see N1) / 73.4732424% and the WTP coefficients re-derived (0.078863062·E ✓). Note: the 05.06.26 .dta embeds the 4764-based WTP_calculated; the doc's re-run categorical regression nevertheless reproduces bit-for-bit from the .dta with the new coefficients (slope .5439353, intercept −.691843, dummies .2671588/.5057716/.9411747/1.822471 — all verified).
- **W2** — derivation corrected: √0.0010186 = 0.0319155 ✓ (summary-table cell still stale — see W2r).
- **W4** — C H&S SE now 0.271249 ✓ (z 0.6023, p 0.547).
- **W5** — Udo-Imeh t now 5.009 ✓.
- **F1** — flexibility IVW summary table now carries its own weights (no longer RT's pasted column).
- **F2/F3** — cross-trait w_N = 0.14947437, w_A = 0.27144591, consistent with the table's Σ1/SE² = 4708.16127478 ✓ (Σ still contains F10's wrong C entry 757.43875).
- **F4/F5** — final IVW coefficients recomputed and internally consistent (−0.05389019·N, +0.047388·A with the sign fixed ✓); N still carries F8, and all five still carry F10.
- **F6** — H&S E coefficient now 0.15827863 × 0.1501833 = 0.02377081 ✓ (multiplied, not added).
- **F7** — Latzman N weight corrected (1/0.1490775² = 44.996; N pool weights now 48.17/36.21/6.40/9.21% ✓).
- **F15** — anchored flexibility code now `0.25* z_stdactions + 0.75 * z_Flexibility_calculated_ivw` ✓.
- **G1/G2** — the malformed figure labels ("0.0.0237", income→RT 0.0077) no longer occur anywhere in the document text (figure apparently replaced/removed; the remaining media are binary EMF images with no such strings).
