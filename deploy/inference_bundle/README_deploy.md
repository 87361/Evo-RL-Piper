# Inference Bundle Deploy Guide

## Purpose
Copy this folder into TeleManipulation and call `inference_runner.load_policy()` and
`inference_runner.predict_action()`.

## Runtime dependency boundary
- This inference bundle is self-contained for runtime.
- OpenPI / APO / RLinf / TeleManipulation repository paths are development references only.
- They are NOT runtime prerequisites for this copied bundle.

## Environment split (important)
- Training repository environment uses `pyproject.toml` + `uv.lock`.
- Copied inference bundle runtime uses only local `requirements.txt`.
- Do not assume training dependencies are available inside TeleManipulation runtime.

## Quick start
1. Copy this whole directory into TeleManipulation repository.
2. Confirm mounted TOS path in `weights_locator.yaml`.
3. Install minimal dependencies: `pip install -r requirements.txt`
4. In TeleManipulation runtime:
   - `from inference_runner import load_policy, predict_action`
   - `policy = load_policy(bundle_dir="deploy/inference_bundle")`
   - `action = predict_action(policy, obs_state)`
5. If default TOS mount path is unavailable, pass an explicit path:
   - `policy = load_policy(bundle_dir="deploy/inference_bundle", weight_path="/path/to/model.npz")`

## Weight policy
- Weights are NOT shipped in code repository.
- Runtime should read from mounted TOS path by default.
