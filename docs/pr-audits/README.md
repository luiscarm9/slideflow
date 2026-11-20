# Slideflow Fork — Upstream Pull Request Audit & Verification Ledger

This ledger tracks the systematic audit, conflict resolution, automated GitHub Actions CPU CI, and **NHR@FAU Helma NVIDIA H100** GPU batch verification of all pending upstream pull requests from [`slideflow/slideflow`](https://github.com/slideflow/slideflow/pulls).

| Seq | Upstream PR | Author | Audit Branch | Status | Fixes / Notes | Helma H100 Slurm Job |
| :---: | :--- | :--- | :--- | :---: | :--- | :---: |
| 0 | Infrastructure | `luiscarm9` | `chore/ci-and-hpc-harness` | Merged ([PR #1](https://github.com/luiscarm9/slideflow/pull/1)) | Modern GitHub Actions CI, `importlib.metadata` + `protobuf<5` packaging, Pandas 3.0 & Libvips TIFF fixes, Helma Apptainer `.sif` | SIF `1402993` OK / H100 `1402994` Queued |
| 1 | [#466](https://github.com/slideflow/slideflow/pull/466) — Slideflow 3.1.0 stability + H-optimus-0 | `jamesdolezal` | `pr-audit/466-v3.1.0` | In Review ([PR-466.md](PR-466.md)) | Core 3.1.0 baseline, NumPy 2.x & Python 3.12 CI unlocked, `importlib.metadata` + defensive plugins, H-optimus-0 & ROI tests | Queued |
| 2 | [#450](https://github.com/slideflow/slideflow/pull/450) — Update `seg_utils.py` | `luiscarm9` | `pr-audit/450-seg-utils` | Queued | Segmentation utility improvements | — |
| 3 | [#422](https://github.com/slideflow/slideflow/pull/422) — H-optimus-0 & DinoBloom | `ldonle` | `pr-audit/422-dinobloom` | Queued | Port `DinoBloom` atop `#466`; fix `img_size=224` & `dump_config(model_size)` | — |
| 4 | [#441](https://github.com/slideflow/slideflow/pull/441) — Ordinal classification MIL | `matte-esse` | `pr-audit/441-mil-ordinal` | Queued | Ordinal loss/metrics for MIL | — |
| 5 | [#444](https://github.com/slideflow/slideflow/pull/444) — Multimodal MIL models | `ldonle` | `pr-audit/444-multimodal` | Queued | Multimodal MIL architecture & forward/train tests | — |
| 6 | [#447](https://github.com/slideflow/slideflow/pull/447) — Survival Analysis for MIL | `matte-esse` | `pr-audit/447-mil-survival` | Queued | MIL Cox/NLL survival training & concordance evaluation | — |
| 7 | [#408](https://github.com/slideflow/slideflow/pull/408) — `torch` CoxProportionalHazard | `jamesdolezal` | `pr-audit/408-torch-cph` | Queued | Rebase conflicted CPH branch onto 3.1.0 & align survival conventions | — |
| 8 | [#410](https://github.com/slideflow/slideflow/pull/410) — Balanced bags | `jamesdolezal` | `pr-audit/410-balanced-bags` | Queued | Rebase conflicted MIL bag-balancing feature onto updated MIL module | — |
| 9 | [#463](https://github.com/slideflow/slideflow/pull/463) — Dual magnification tile extraction | `CostanzaSiniscalchi` | `pr-audit/463-dual-mag` | Queued | Evaluate `tfr_mmag` + `#463` combined branch & multi-mag extraction tests | — |
