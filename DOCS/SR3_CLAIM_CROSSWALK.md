# Public v1.0.3 crosswalk note

The printed SR3 numerical table checks have been re-run successfully against this **annotation-redacted, code-reduced public edition**. The historical MCP1 DeiT table previously found inside a nested legacy archive is now preserved byte-exactly as `DATA_TABLES/public_deit_legacy_state_parameter_readouts.csv`; no legacy execution code is necessary for the 284-cell supplement comparison. The following historical audit text describes the original larger source archive and is retained as scientific provenance, **not** a literal current-file layout description. For current execution and rights limitations see `README.md` and `PUBLIC_DATA_PROVENANCE_AND_REPRODUCIBILITY.md`.

---

# Manuscript v3.3 SR3 FINAL and Supplement: numerical source crosswalk

**Source scientific text:** *Opposite Directional Orderings of Targeted Rotary-Frequency Perturbations in DeiT and Swin*; supplied final SR3 manuscript and Supplement, dated 2026-11-07.

**Audit:** complete preserved V33 archived-CPU reproduction finished with `ADSV3_V33_FINAL_ARCHIVED_CPU_REPRODUCTION_PASS` on a clean extraction of v1.1; additional recalculation of 113 independently specified headline/statistical values passed. An independent comparison of 284 printed numeric cells across **9** supplemental LaTeX table files against raw V33 CSV and older embedded historical result tables passed within stated display rounding. Neither count is a claim that *every possible number* in the paper can be regenerated from foundational experiment inputs.

## Reproduction levels and exact sources

| Article / supplement result group | Archived source and executed check | Level |
|---|---|---|
| Public DeiT common-beta and Swin preference by scale; sign counts; exact sign-flip p and inverted confidence bounds; main directional figures | `DATA_TABLES/deit_common_beta_72_state_q.csv`, `DATA_TABLES/directional_swin_72_original_q_states.csv`, `scripts/verify_sr3_research_claims.py`; original DeeT regularization-sensitivity numeric table 36/36 cells checked | **Recomputed from individual saved state-level readouts.** Does not repeat optimizer/GPU inference |
| Historical nonuniform DeiT public scale inference and Swin control comparisons | The DeiT MCP1 original state-readout CSV within `ARCHIVES/legacy/HISTORICAL_NUMERICAL_DATA_ONLY.zip` (itself a ZIP-in-ZIP), plus public Swin 72-state CSV; 48 printed means/p-values checked in `Table_Public_Exact_Contrasts.tex` | **Recomputed from archived source readouts** |
| Conditional arrangement-null `z` table and 186-state chi-square/KS calibration | `DATA_TABLES/deit_common_beta_72_state_q.csv`, `DATA_TABLES/directional_swin_72_original_q_states.csv`, `DATA_TABLES/finite_null_analytic_sigma_186_states.csv`; 186 mean chi2=77.615642, KS D=0.036189, nominal p=0.960597 | **Recomputed from archived state-level sigmas and empirical SD summaries**; not an independent rerun of all original random-null draws or choice of null |
| Twelve separately trained Base-checkpoint directional effects; 6+6 exact seed tests, CIs, sign and SD | `DATA_TABLES/trained_base_288_state_q.csv`, 113 claim checks and 36 printed six-seed table cells | **Recomputed from saved state readouts**; checkpoint training and original attack optimization not rerun |
| 144 S5 frozen attack states, 72 paired damage contrasts, 7.2 million classification records, per-seed differences | `ARCHIVES/s5/S5_FULLVAL_144_STATES.zip`, `ARCHIVES/s7/S7_TRAINED_CLEAN_FULLVAL.zip`, S5 state/cohort/seed CSV, `scripts/verify_v33_raw.py` and complete CPU runner; 36 printed per-seed table cells | **Recomputed from saved prediction arrays**, not new GPU inference or fresh attack generation |
| S7 12 trained + 2 public Base clean-accuracy streams, sample SD and comparison | `ARCHIVES/s7/S7_TRAINED_CLEAN_FULLVAL.zip`, `ARCHIVES/s7/S7_PUBLIC_CLEAN_FULLVAL.zip`, `DATA_TABLES/S7_TRAINED_VS_PUBLIC_12_PAIRS.csv`; 36 printed per-seed cells | **Recomputed from saved NPZ predictions and labels**; not new image inference |
| A6 14 models, six 256-image cohorts, 1,536 distinct identities, 777 class labels, 168 displacements, 84 paired contrasts | `ARCHIVES/a6/A6_ORIGINAL_168_RAW.zip`, verified F-vector ZIP `A6_DIRECTIONAL/ADSV3_14_F_VECTORS_VERIFIED_20261106.zip`, `A6_DIRECTIONAL/recompute_a6_f.py`, six CSV comparisons; 42 printed model rows | **Recomputed from archived F vectors and perturbation tensors**; their generation from images not rerun. A6 frequency analysis is retrospective |
| A6 x/y additive and Swin four-stage contributions | A6 paired/stage CSVs and raw A6 replay; exact main text and supplemental stage summaries checked | **Recomputed archived-tensor projections** |
| Six-cohort DeiT target-locality intervention and descriptive attenuation | `DATA_TABLES/deit_locality_six_cohorts.csv`, 18 printed cohort cells, exact sign flip and bound in 113 checks | **Recomputed from saved cohort summaries**; target optimization not rerun |
| Trained-checkpoint S6 first-gradient directions | `DATA_TABLES/trained_checkpoint_first_gradient_288_states.csv`, `ARCHIVES/s6/S6_288_GRADIENT_METRICS.zip`; selected means/signs checked | **Derived from saved initial-gradient metrics**; original gradient tensors and live backward passes are not regenerated |
| Supplement table of nine public reference-frequency geometries and endpoint feasibility | `Table_Reference_Geometry.tex` contains numeric summary; package contains **14 Base F vectors**, not the original public Small/Large/Tiny F tensor bytes for all nine reference configurations | **Not fully independently regenerable from this ZIP alone**. Published summary retained; would require original 9 reference F tensors/models |
| Supplement pooled fixed-public first-sign-gradient table (`Table_Initial_Sign_Gradient.tex`) | Historical public experiment artifacts exist in nested legacy archives; this audit did **not** independently regenerate every entry and all first-step confidence bounds from original gradient tensors | **Only partial source-level coverage; not a complete verified table** |
| Main text and supplement figures (six main PDFs, three supplementary PDFs) | Numerical source rows exist for central curves, and A6 supplies `make_figures.py`; original figure PDFs are in the **separate scientific-text archive** | **Plot data traceable**, but all nine PDF figures were **not independently regenerated and visually compared** here |
| Model architecture, hyperparameters, 400+20 / 300 training epochs, final checkpoint SHA/lineage | Historical configs and provenance references in this archive and the privately preserved Step-9 repository | **Documented method/provenance; not reproducible from this ZIP as new training** |

## Direct supplementary LaTeX table-to-source numeric check

`python scripts/verify_sr3_table_cells.py /path/to/extracted/NEUROCOMPUTING_ADSV3_V33_SR3_SCIENTIFIC_TEXT`

The audit checks **284/284 displayed numeric cells** in nine table source files (the printed source values are compared against newly computed means or underlying exact archived records at their stated rounding precision):

| Table file | Checked cells |
|---|---:|
| Table_Public_Exact_Contrasts.tex | 48 |
| Table_DeiT_Regularization_Sensitivity.tex | 36 |
| Table_DeiT_Target_Locality.tex | 18 |
| Table_Training_Seed_Contrasts.tex | 36 |
| Table_Secondary_Checkpoint_Results.tex | 8 |
| Table_Full_Validation_Damage.tex | 36 |
| Table_Public_Clean_Accuracy.tex | 36 |
| Table_Public_Standardized_Alignment.tex | 24 |
| Table_Image_Disjoint_Directional_Results.tex | 42 |
| **TOTAL** | **284** |

The remaining three supplemental table files include: (1) reference frequency geometry, (2) public initial sign-gradient diagnostics, and (3) trained initial-gradient checkpoints. The third has its **five principal means** checked directly by `verify_sr3_research_claims.py` but has not been included in the 284 source-table-cell count. No incorrectly implied 12/12 exhaustive table audit.

## Runbook

1. On a **fresh, trusted extraction** of the archived-CPU ZIP, install `numpy`, `torch`, `scipy` into an isolated Python environment.
2. Run `python scripts/run_all_cpu.py` to reproduce the complete archived numerical suite and the extra 113 SR3 claim checks. Success markers include `SR3_SELECTED_NUMERICAL_CLAIMS_PASS` and `ADSV3_V33_FINAL_ARCHIVED_CPU_REPRODUCTION_PASS`.
3. For the direct supplemental table-value comparison, extract the **scientific-text ZIP** separately and run the `verify_sr3_table_cells.py` command above. Expect `SR3_SUPPLEMENT_TABLE_CELLS 284 PASS 284 FAIL 0`.
4. Inspect `OUTPUT/SR3_NUMERICAL_AUDIT.json` and `OUTPUT/SR3_TABLE_CELL_CROSSCHECK.json` for traceable calculated-versus-published numbers.
5. For limits, see `DOCS/REPRODUCIBILITY_SCOPE.md` and this table. Raw ImageNet and original 12 weight checkpoints are not bundled.

## Interpreting completeness

The **reported central numerical findings are independently supported by archive-level reanalysis**, including prediction- and A6-tensor-level replay. Nevertheless, the stronger claim that **all scientific results are fully independently reproducible from foundational raw image/model inputs is false** for the present archive. Missing independent regenerations include some reference-frequency geometry, parts of public first-gradient source traces, exact final figure outputs, and any fresh optimization/training/inference. These are reproducibility scope boundaries, not discovered discrepancies in the checked table values.

Licensing and distribution of software code, historic scripts, and ImageNet-derived identifiers/metadata remain separately scoped; the CC BY data notice does not relicense those components.
