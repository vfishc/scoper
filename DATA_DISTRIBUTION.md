# Benchmark data distribution

Project: https://vfishc.github.io/scoper/  
Code: https://github.com/vfishc/scoper  
Paper: [bundled preprint](docs/downloads/SCOPER_preprint.pdf)  
Hugging Face: https://huggingface.co/datasets/vfishc/SCOPER.

## Required task assets

The approximately 22 MiB `task_bank/` tree is bundled in GitHub and prepared as an identical Hugging Face mirror. It contains our task instructions, initial-state/spawn settings, frozen role/asset assignments, QA, Binding sidecars, oracle action references and manifests. The two copies must remain byte-identical. Use the frozen release revision rather than silently mixing downloaded versions.

Each backend has five coupled files: `tasks_with_qa.json`, `binding.json`, `oracle_reference.json`, `asset_requirements.json`, and `manifest.json`. Both bank hashes are verified by `scoper check-bank`.

These are simulator task configurations, not a conventional training dataset. Use the supplied CLI; no automatic `datasets.load_dataset` adapter is claimed. Oracle references, binding internals and QA answer keys belong to the execution/scoring infrastructure, not the policy prompt.

## Optional Hugging Face artifacts

Oracle videos, action-indexed frames and detailed execution traces are suitable for a separate optional download, grouped by backend/template/episode and indexed by task ID and bank hash. They are not yet packaged here. Historical rendered episodes must be checked against this frozen bank before inclusion, especially the replacement inspection tasks. They are not required for direct VLM evaluation.

## Obtain from original publishers

Do not bundle official BEHAVIOR object meshes/scenes/robot assets, Isaac Sim, AI2-THOR binaries, ProcTHOR's official layout pool or model weights. The README provides upstream setup/download instructions. Task asset requirements identify the required scene/object models; they do not include their geometry.

## Publishing

The Hugging Face dataset is hosted at `vfishc/SCOPER`. Keep GitHub and HF release manifests aligned.

Original code is MIT-licensed; SCOPER-authored task-bank data is CC BY 4.0. See [license scope](licenses/README.md) and [data attribution](task_bank/ATTRIBUTION.txt). External simulator assets, code, paper PDFs and visual media are not automatically relicensed.
