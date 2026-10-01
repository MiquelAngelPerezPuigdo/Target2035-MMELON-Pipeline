# Challenge workspace recovery

On October 1, 2026 the local campaign was copied from the Codex Cache Challenge workspace into `local_del/` in this Git repository. The original workspace and session records remain intact. [asset_inventory.json](recovery/asset_inventory.json) records original and recovered SHA-256 hashes, sizes, and deliberate portability changes.

## Preserved assets

Source code, reports, all 21 original validation ZIP archives, per-round available manifests, and selected candidate details are versioned. The evaluated feedback ledger is preserved unchanged, including its historical inconsistencies. [recovery_status.json](recovery/recovery_status.json) records corrections separately. Compact derived validation labels and the 100 selected test negatives are available under `docs/recovery/` for next-phase ingestion, with evidence-source labels.

Prepared data, features, model weights, raw external reference files, full test CSVs, and the local LightGBM runtime were copied and remain ignored. The historical ChemBERTa Python environment and the IBM vendor source were left at the original location; environments should not be relocated by copying their executables. The old environment is `~/.codex/.chatgpt-projects/g-p-6aa392213d008191a42dc240e16bc733/local_del/mmelon_env/`.

Four active scripts were changed to resolve input files relative to the repository rather than a hard-coded home path. The historical runner resolves its Python interpreter and Torch runtime from the local environment. Frozen round-three code and original model files were not changed.

The root MMELON README and PLAN were moved without changing their contents to `docs/archive/`. Their claims about training speed, combinatorial information content, and OOD performance were hypotheses, not evidence from this campaign. The untracked historical `deploy.sh` is now preserved in Git as part of that earlier effort; recovery did not execute it or submit an HPC job.

## Restore assets after cloning

On the original Mac, copy the preserved local artifacts from the Codex workspace:

```sh
rsync -a --exclude='__pycache__' --exclude='.DS_Store' \
  "$HOME/.codex/.chatgpt-projects/g-p-6aa392213d008191a42dc240e16bc733/local_del/data/" local_del/data/
rsync -a --exclude='__pycache__' --exclude='.DS_Store' \
  "$HOME/.codex/.chatgpt-projects/g-p-6aa392213d008191a42dc240e16bc733/local_del/features/" local_del/features/
rsync -a --exclude='__pycache__' --exclude='.DS_Store' \
  "$HOME/.codex/.chatgpt-projects/g-p-6aa392213d008191a42dc240e16bc733/local_del/models/" local_del/models/
rsync -a --include='Team_*.csv' --include='*/' --exclude='*' \
  "$HOME/.codex/.chatgpt-projects/g-p-6aa392213d008191a42dc240e16bc733/local_del/submissions/" local_del/submissions/
```

Also restore the local `vendor/lightgbm` runtime or install the recorded LightGBM version in a compatible environment. Obtain the challenge raw data and validation/test input files through their original authorized source; these were already present in this repository's ignored directories. Exact hashes and paths are listed in the inventory. A new machine cannot recover private large assets from Git alone.

## Verification and limitations

Run `local_del/verify_recovery.py` as described in the root README. It checks all validation lists, label constraints, full test files, frozen hashes, scaffold holdouts, fingerprint parity, and saved-model replay. The verification output is saved under `.local/` to avoid overwriting original reports; the reviewed recovery receipt is in `docs/recovery/verification.json`.

The older `verify.py` expects a generated round-one directory and manifest that are no longer present in the original workspace. The original round-one ZIP survives, so the recovery verifier reads it directly. It does not regenerate missing evidence or pretend the old verifier still passes unchanged.

Some saved START_HERE files still say not submitted and contain old quota estimates. They are historical snapshots, not current instructions. Evaluation status comes from the recovered feedback ledger and chat history. Submission IDs, current account quota, and current released-label filenames remain unresolved.
