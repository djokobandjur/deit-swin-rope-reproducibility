# DeiT and Swin RoPE Research Resources — Reproducibility

**Public source-code and tabular-results companion** for the peer-review research resources of Đoko Banđur and Miloš Banđur. This GitHub repository is a **clean source-only companion** to a complete, publicly deposited numerical reproducibility archive on Zenodo.

## Published and citable research data

**Banđur, Đoko, and Banđur, Miloš (2026). _DeiT and Swin RoPE Research Resources: Reproducibility Archive_, version 1.0.3. Zenodo.**

- **Version-specific DOI:** https://doi.org/10.5281/zenodo.24055154
- **All-versions DOI:** https://doi.org/10.5281/zenodo.24055153
- **Full archived numerical replay:** download the published 48.3 MB ZIP from Zenodo; SHA-256 of the exact deposited source ZIP: `b7686dffa07e90c47ec6eb99edcc213e361226efee6ae548f21ee42440c6d40f`.
- **Model weights:** 12 independently trained DeiT/Swin RoPE model checkpoints are maintained separately on Hugging Face under BSD 3-Clause; this repository does not include them.

## What is in this GitHub repository?

- **12 author-developed numerical/CPU verification scripts** (`scripts/` and `A6_DIRECTIONAL/`), with **MIT** licensing scoped by `CODE_LICENSE_SCOPE.md`.
- **Original numerical summary tables and author-created interpretive metadata**, with **CC BY 4.0** licensing scoped by `LICENSE_DATA.md`.
- Claim-to-result crosswalk, citation metadata, upstream methodology acknowledgments, and exact-source checksums.

The GitHub checkout **does not contain** the much larger saved S5/S7 prediction streams, 168 A6 perturbation tensors, 14 reference F vectors or other nested raw numerical ZIPs. Those are distributed in the version-locked **Zenodo** archive. Therefore, **do not expect `python scripts/run_all_cpu.py` to pass on this GitHub checkout alone.** It must be run from the extracted full Zenodo reproducibility archive; it uses `ARCHIVES/...` inputs absent here.

### Reproduce from the complete archived numerical sources

1. Open https://doi.org/10.5281/zenodo.24055154 and download the archive file.
2. Verify its file integrity with the SHA-256 above.
3. Extract the Zenodo ZIP to a local directory and open the top-level `DeiT_Swin_RoPE_Reproducibility_Archive_v1.0.3/` folder.
4. Follow `TESTED_CPU_ENVIRONMENT.txt` and run `python scripts/run_all_cpu.py` from that directory.
5. For paper-level numeric verification use `scripts/verify_sr3_research_claims.py` and compare the saved supplementary checks in `DOCS/SR3_TABLE_CELL_CROSSCHECK.json`.

The reported archive-level CPU replay completes successfully and checks 113/113 selected numeric/statistical claims and 284/284 displayed numeric cells from nine supplementary tables. This is **archived numerical reproducibility**, not new training or fresh inference on ImageNet images. Selected original public-reference frequency and gradient sources, and complete figure regeneration, have documented limitations.

This GitHub source-only release tag can link to the same Zenodo data DOI; it does **not** mint or claim a second Zenodo DOI. GitHub's automatically generated tag-source archive is not a substitute for the full Zenodo dataset.

## Data provenance and rights

Original data and commentary: **CC BY 4.0** (`LICENSE_DATA.md`); the **12 explicitly scoped original scripts**: **MIT** (`LICENSE_CODE.md` and `CODE_LICENSE_SCOPE.md`). Third-party source and ImageNet ground-truth/identity annotations are **not** sublicensed. Full public Zenodo archive omits the original ImageNet ground-truth arrays and certain historical source code, while preserving authors' saved predictions, correctness outcomes and archival tensors. Cite original RoPE-ViT/DeiT/Swin and ImageNet publications as appropriate.

## Citation

```bibtex
@dataset{bandjur2026deitswinrope_repro,
  author    = {Banđur, Đoko and Banđur, Miloš},
  title     = {DeiT and Swin RoPE Research Resources: Reproducibility Archive},
  year      = {2026},
  version   = {1.0.3},
  publisher = {Zenodo},
  doi       = {10.5281/zenodo.24055154},
  url       = {https://doi.org/10.5281/zenodo.24055154}
}
```

Formal scholarly citation is recommended, while legal attribution follows the applicable file-level license. Always identify the exact HF checkpoint revision and SHA-256 where a particular model is used.
