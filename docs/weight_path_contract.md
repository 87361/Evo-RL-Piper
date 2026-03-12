# Weight Path Contract

## Goal
Define a stable weight path convention for TeleManipulation deployment without storing weights in this repository.

## Path Rule
- `weights_locator.yaml` must include `policy_weights_path`.
- The path should point to the mounted TOS artifact location, for example:
  - `/mnt/tos/openpi_v0/run_001/model.npz`
- Exported bundle may include `local_debug_fallback` for local debugging only.

## Runtime Resolution
`inference_runner.load_policy()` resolves weight path in this order:
1. Explicit `weight_path` argument.
2. `policy_weights_path` from `weights_locator.yaml`.
3. `local_debug_fallback` from `weights_locator.yaml`.

## Delivery Principle
- Bundle code can be copied.
- Weight file is not copied into git repository.
- Runtime path is configurable but function signature remains stable.
