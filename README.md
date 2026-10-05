# DEL ML Challenge PGK2

This is the consolidated workspace for the DREAM x CACHE Target 2035 DEL-ML Challenge. The September 2026 local experiments and their feedback have been recovered from the Codex Cache Challenge project. Start with the [phase-one report](docs/PHASE1_REPORT.md) and [Active Learning plan](phase2/README.md).

The full Results page history contains 51 phase-one validation submissions: 19 zero-hit and 32 nonzero submissions, totaling 163 hits. The strongest validation panel found 25 hits in three chemical series. Both blind-test submissions found zero selected hits. In Active Learning, three phase-two validation panels have also scored zero; the audit found 29 previously known negatives in the DEL-only panel, no known validation positives in any current panel, and no score-to-ID mapping error.

## Where everything lives

| Location | Contents |
| --- | --- |
| [docs/PHASE1_REPORT.md](docs/PHASE1_REPORT.md) | Methods, results, limitations, and lessons |
| [docs/CHAT_HISTORY.md](docs/CHAT_HISTORY.md) | Recovered conversation history and corrected assumptions |
| [docs/PHASE1_RESULTS.csv](docs/PHASE1_RESULTS.csv) | All 48 recorded validation sets and two evaluated test sets |
| [phase2/reports/validation_history.json](phase2/reports/validation_history.json) | Synapse IDs, names, and scores for all 51 phase-one and three current validation submissions |
| [phase2/reports/validation_panel_history_overlap.json](phase2/reports/validation_panel_history_overlap.json) | Current panel overlap with historical submissions and forced labels |
| [phase2/reports/validation_row_provenance_audit.json](phase2/reports/validation_row_provenance_audit.json) | Exact check from selected CatalogIDs through structures, features, and model scores |
| [phase2/reports/validation_release_transfer.json](phase2/reports/validation_release_transfer.json) | Chemistry overlap between released binders and forced historical validation labels |
| [phase2/reports/validation_panels.json](phase2/reports/validation_panels.json) | Phase-two model comparison, panel provenance, and current versus historical panel ranks |
| [local_del/](local_del/) | Local DEL models, rounds 2–21, audits, result ledger, and submissions |
| [docs/recovery/](docs/recovery/) | Source inventory, hashes, and recovery verification |
| [phase2/](phase2/) | Active Learning plan and separate campaign status |
| [docs/archive/](docs/archive/) | Original MMELON README and strategy, preserved as historical proposals |
| Root Python scripts and `submit_euler.sh` | Earlier MMELON and HPC approach |

The original local experiments remain at `~/.codex/.chatgpt-projects/g-p-6aa392213d008191a42dc240e16bc733/local_del/`. The recovered copy here is the working project. Full datasets, model weights, feature caches, and full test CSVs are present locally and ignored by Git. A fresh clone contains the compact evidence and code; it does not include those large assets. [Recovery notes](docs/RECOVERY.md) explain how to restore them.

## Check the recovered work

On this Mac, use the existing environment and vendored LightGBM runtime:

```sh
TORCH_LIB=$(venv/bin/python -c 'import importlib.util; from pathlib import Path; print(Path(importlib.util.find_spec("torch").origin).parent / "lib")')
DYLD_LIBRARY_PATH="$TORCH_LIB" venv/bin/python local_del/verify_recovery.py
```

The check validates original validation archives, logically forced labels, the two evaluated test files, frozen hashes, scaffold separation, and saved-model predictions. It does not train models, submit files, or overwrite the phase-one ledger.

`local_del/run_local.sh` is the historical round-one rebuild entry point. It overwrites generated assets; use an isolated copy for a rerun. Source inputs default to this repository and can be set with `DEL_SOURCE_ROOT`; its interpreter can be set with `DEL_PYTHON`. The base package versions are recorded in [requirements-base.txt](local_del/requirements-base.txt). ChemBERTa needs the separate historical `mmelon_env`, retained in the original Codex workspace.

## Current phase

As checked on October 5, 2026, the official [timeline](https://www.synapse.org/Synapse:syn75349604/wiki/641065) lists Active Learning for October 1–31, with 100 validation credits after the additional data release and two test submissions. The actual account balance still needs checking. The old phase-one quota estimates must not be used as today's balance.

The October 5 [release audit](phase2/DATA_AUDIT.md) confirms 6,580 experimental labels, including 39 binders, and an updated template with 178,052 eligible test IDs. The raw files and prepared tables are saved locally under ignored `phase2/data/`. Phase-two models were compared and three 50-ID validation panels were submitted; all scored zero. The full phase-one feedback shows the earlier campaign had strong ligand-informed results, so do not infer that every method failed. Use the [campaign status](phase2/status.json), [first experiment report](phase2/EXPERIMENT_01.md), and mapped [validation history](phase2/reports/validation_history.json) before choosing another panel.

The [October 5 site review](phase2/SITE_REVIEW.md) records current organizer clarifications and the first experiment sequence. All 6,580 released labels now have fresh features from the official released structures, including the eight flagged constitutional differences. Phase-two models were compared and three 50-ID validation panels were submitted; all scored zero. The full phase-one feedback shows the earlier campaign had strong ligand-informed results, so do not infer that every method failed. Use the [campaign status](phase2/status.json), [first experiment report](phase2/EXPERIMENT_01.md), and mapped [validation history](phase2/reports/validation_history.json) before choosing another panel.
