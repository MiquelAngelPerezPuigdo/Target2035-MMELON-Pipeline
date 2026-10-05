# Recovered local PGK2 experiments

This directory contains the September 2026 phase-one campaign recovered from the Codex Cache Challenge project. Start with the consolidated [phase-one report](../docs/PHASE1_REPORT.md), [chat history](../docs/CHAT_HISTORY.md), and [Active Learning plan](../phase2/README.md).

## Code and evidence

`prepare.py`, `features.py`, `train.py`, `similarity.py`, and `export_round.py` implement the initial DEL-supervised local workflow. `round02.py` through `round21.py` preserve later adaptive experiments. The ErG, ChemBERTa, shape, kernel, and aggregate-count scripts preserve their corresponding experiments.

`reports/official_results.json` records the user-reported feedback and now includes the Synapse submission IDs matched from the Results page. The complete score-to-file mapping, including three early submissions outside rounds 01–21, is in [validation_history.json](../phase2/reports/validation_history.json). Its quota fields are historical and contain a corrected mistake; use [recovery_status.json](../docs/recovery/recovery_status.json) for that correction. `submissions/roundNN_validation_batch.zip` contains each original batch. Several extracted directories were missing in the original workspace, but all 21 batch archives survived.

The evaluated final tests are `submissions/blind_test/Team_MMELON_T1_FROZEN_G.csv` and `submissions/blind_test_revised/Team_MMELON_T2_REVISED_COUNTS.csv`. The original diverse T2 was replaced before submission. Both evaluated files returned zero selected hits. Old START_HERE files describe preparation-time status and are not current submission instructions.

## Replay and verification

Use `verify_recovery.py` through the root README command to verify the copied data and original models. `--compact` checks the validation archives and derived labels using only Git assets. It needs no scientific packages. The historical `verify.py` expects a missing round-one manifest, so the recovery verifier reads the surviving original ZIP instead.

`requirements-base.txt` records the existing base environment. LightGBM is available locally in ignored `vendor/`. ChemBERTa used the separate `mmelon_env` retained in the original project. `run_local.sh` rebuilds the initial pipeline and overwrites generated outputs; use an isolated copy if reproducing training.

Large local data, features, model weights, and full prediction CSVs are ignored and indexed by hash. Preserve this phase-one directory as evidence and put new experiments in `phase2/`. The original directory README is archived unchanged at [LOCAL_DEL_README.md](../docs/archive/LOCAL_DEL_README.md).
