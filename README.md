# BEV-JA

Source code of the experiments for the manuscript
**"BEV-JA: Goal-Aligned Joint Road-Patch Attacks on Object Detection and Monocular Depth Estimation"**
(manuscript in preparation).

BEV-JA optimizes one adversarial patch on the road in front of a vehicle so that, for the same vehicle,
an object detector misses it and a monocular depth estimator places it farther away.

## What this repository is

This is the research code as it was executed, not a packaged library.

- Every Python file is a byte-identical copy of the file on the experiment machine (taken on 2026-11-12), with the
  six exceptions listed under "Strings replaced for publication". `SOURCE_MANIFEST.json` lists the SHA-256 of each file.
- It contains the method, the evaluation, the joint-attack-success-rate (JASR) statistics and the configuration.
- It does **not** contain datasets, model weights, optimized patches or result files.
- The code that produced the baseline results in the manuscript is not included.

## Where the method is

| Component in the manuscript | File |
|---|---|
| One-sided far-depth margin objective | `packages/science/depth_hinge_objective.py` |
| Detection deficit and target matching | `packages/science/detection_control.py` |
| Progress-aware controller (period, model weights, task priorities, EMA) | `packages/science/periodic_controller.py` |
| Gradient normalization, conflict constraint, norm matching | `packages/science/task_gradient_balance.py` |
| Shared geometric views (identity + one random crop / scale / shift / rotation) | `packages/science/geometry_eot.py` |
| Per-scene optimization loop | `packages/runner/derived_budget_training_engine_v2.py` (`run_sample_optimization`) |
| Stable Diffusion 3.5 Medium step used as road-texture prior (two CLIP text encoders, T5 disabled) | `packages/runner/sd35_conditional_two_clip_v1.py` |
| Road-plane patch projection and patch file format | `packages/base/parent_v17/03_方法/production/road_patch_contract.py` |
| Vehicle cut-out insertion and scene construction | `packages/base/parent_v17/01_公共数据集/advrm_insertion.py`, `packages/base/parent_v17/03_方法/production/road_runtime.py` |
| Detector and depth-model adapters | `packages/base/parent_v17/02_模型/model_adapters.py` |
| Evaluation on four detectors and four depth models | `packages/evaluation/evaluate_ablation_full8.py`, `packages/evaluation/eight_target_library.py` |
| Det-ASR / Dep-ASR / JASR / Far-Shift | `jasr/jasr_from_evaluation_dir_v1.py`, `jasr/jasr_core.py` (thresholds in `jasr/atomic_counts.py`) |

## Layout

```
packages/science        method modules (objectives, controller, gradient coordination, EOT, contracts, audits)
packages/runner         training entry point and optimization engine of the final configuration
packages/runner_variants/b3_tp10   override that admits controller period 10
packages/evaluation     evaluation on eight target models and its CPU re-verification
packages/base           rendering, scene construction, model adapters, metric code; an earlier package of the project that the
                        method binds as a whole, so it also holds earlier development modules (directory names kept as on the machine)
packages/clip_source, packages/r2_source, packages/v2_source   earlier modules that the method imports
packages/ops            process helper used by the runner
jasr                    success-rate statistics from a retained evaluation directory (CPU only)
queue                   sequential driver that builds the requests and runs train -> check -> evaluate -> verify
protocol                configuration, the 200-scene selection, the 100 nuScenes scenes, hyper-parameter variants
```

## Configuration

`protocol/final_configuration.json` is the configuration of the reported BEV-JA runs: 5 outer rounds of 25 latent
updates per scene (125 updates), identity view plus one random shared view per update, controller period 5,
`model_mu` 0.5, `task_mu` 0.8, EMA 0.7, depth margin 0.1, latent learning rate 0.05, seed 20260919.
It is extracted without change from `protocol/PROTOCOL.json`.

`protocol/FIXED200_SELECTION.json` lists the 200 background images (100 KITTI, 100 nuScenes mini) by file name and SHA-256.
`protocol/hyperparameter_variants.json` lists what each run of the sensitivity study changes.

## Which runs this code corresponds to

`packages/runner` and `packages/evaluation` are the packages of the final-configuration run on the 100 nuScenes scenes
(`protocol/nuscenes100_samples.json`). That run gives a 16-pair macro JASR of 4.125 %; the manuscript reports 4.12 % for nuScenes. The hyper-parameter runs use the same packages with the changes in `protocol/hyperparameter_variants.json`.

The 200-scene runs use the same configuration with a four-part plan (50 scenes per part). The runner package of those
runs was not on the machine this copy was taken from and is not in this repository.

## Requirements

Versions installed on the experiment machine are in `requirements.txt` (Python 3.12.3, PyTorch 2.8.0 + CUDA 12.8,
diffusers 0.38.0). The runs of these packages used one NVIDIA RTX PRO 6000 (96 GB).

Not included and needed to run:

- KITTI object-detection images and nuScenes v1.0-mini front-camera key frames
- Stable Diffusion 3.5 Medium weights (gated on Hugging Face)
- Weights of Faster R-CNN, SSD300, FCOS, Monodepth2, DPT-Large, Depth Anything V1 and ZoeDepth
- The YOLOv5 runtime source and `yolov5s.pt` (AGPL-3.0, not redistributed here)

`queue/TEMPLATE_INPUTS_B3_SERVER.json` names every external file the runs bind, with size and SHA-256.

## Running

The code does not run as downloaded.

1. Paths are absolute paths of the experiment machine (`/root/...`).
2. Before it starts, the runner verifies five source packages against their `PACKAGE_MANIFEST.json`
   (`packages/science/source_bindings.py`). Only the Python sources and run-time configuration of those packages are
   published; internal notes, run records and execution plans are left out and listed with their hashes in
   `OMITTED_FILES.md`. The verification therefore fails on this copy.

### Strings replaced for publication

Eight files differ from the executed files in strings only; no statement or value that affects a computation was
changed. `SOURCE_MANIFEST.json` marks them with `strings_replaced` and keeps the hash of the executed file as
`original_sha256`.

- `jasr/detection_only_from_stage_v1.py`, `jasr/jasr_from_evaluation_dir_v1.py`, `jasr/merge_jasr_atomic_v1.py`:
  the `schema` label written into the output file was renamed.
- `queue/run_b3_queue.py`: the runner and schema labels written into its records, one provenance note and the module
  docstring were reworded.
- `packages/science/test_contract200.py`, `packages/base/reference_analysis/analyze_train_updates.py`: the name of the
  marker file used to locate the original workspace was changed to `.workspace_root`. Both files need the original
  workspace and do not run on this copy.
- `protocol/PROTOCOL.json`, `protocol/FIXED200_SELECTION.json`: in ten recorded source paths the workspace prefix was
  replaced by `WORKSPACE_ROOT`. The hashes of these two files that are pinned in the code (for example `PROTOCOL_SHA`
  in `packages/runner`) and in the package manifests are those of the executed files, so those checks fail on this
  copy as well.

`queue/run_b3_queue.py` powers the host off when its queue is finished unless `--no-shutdown` is given.

These unit tests run on a CPU without data or weights (48 tests, one of them skipped; checked on macOS with PyTorch installed):

```bash
cd packages/science   && python -m unittest test_periodic_controller test_task_gradient_balance
cd packages/v2_source && python -m unittest test_depth_spatial
cd packages/r2_source && python -m unittest test_projection_cpu_backward
```

The other test modules need the import paths and files of the original workspace.

## Scope of the experiments

All experiments are digital: a vehicle cut-out is composited into each background image and the patch is rendered
onto the road region of that image. No physical patch was printed or tested. The code is released to make the
reported robustness evaluation inspectable.

## Third-party material and license

See `THIRD_PARTY_NOTICES.md`. No license has been chosen for the authors' own code yet.
