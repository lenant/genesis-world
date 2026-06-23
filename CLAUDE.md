# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

Genesis World is a universal, GPU-accelerated multi-physics engine for physical AI / robotics, exposed through a Pythonic `import genesis as gs` interface. `AGENTS.md` and `.github/contributing/` (ARCHITECTURE, CODING_CONVENTIONS, TESTING, PULL_REQUESTS, EXAMPLES, USD_PARSER) hold the canonical contributor docs — read them for detail; this file captures the load-bearing big picture and the gotchas.

## Setup, build, test

PyTorch is **not** a declared dependency — it must be installed separately and platform-specifically *after* the package, or imports fail (see `genesis/__init__.py`).

```bash
uv sync
uv pip install torch --index-url https://download.pytorch.org/whl/cu126  # cu126 / cpu / (bare = Apple Metal)
```

```bash
# Tests (pytest defaults to CPU; runs parallel via xdist, excludes benchmarks+examples)
uv run pytest tests/                       # full suite
uv run pytest tests/test_rigid_physics.py  # single file
uv run pytest tests/test_x.py::test_y      # single test
uv run pytest tests/ -m required           # minimal must-pass set (run before any PR)
uv run pytest tests/ -m "not slow"         # skip >100s tests
uv run pytest tests/ --backend=gpu         # GPU backend
uv run pytest tests/ --vis                 # interactive viewer (disables parallelism)
uv run pytest tests/ --dev                 # genesis debug mode
uv run pytest tests/ -p no:pytest-retry -p no:rerunfailures   # in sandboxes/containers

# Lint/format — ruff, 120 cols, via pre-commit (auto-formats genesis/, excludes genesis/ext/)
pre-commit install
pre-commit run --all-files
```

Markers (`pyproject.toml`): `required`, `slow`, `examples`, `benchmarks`. The `examples`/`benchmarks` marks are excluded from the default run. Test config and fixtures (`initialize_genesis`, `backend`, `precision`, `tol`, `show_viewer`) live in `tests/conftest.py`.

PR titles are prefixed: `[BUG FIX]` / `[FEATURE]` / `[MISC]` / `[CHANGING]` / `[BREAKING]`.

## Architecture

The user-facing flow is a layered build-then-step pipeline:

```
gs.init(backend=...) → gs.Scene → Simulator → Solvers → Entities
                                       ↓
                                  Visualizer → Viewer / Cameras (Nyx | Luisa | Pyrender)
```

- **`gs.init()`** (`genesis/__init__.py`) sets global state (`gs.device`, `gs.backend`, `gs.logger`, `gs.EPS`). Backend auto-selects CUDA→ROCm→Metal→CPU unless pinned; `debug=True` forces CPU. `gs` is a heavily stateful singleton module — there is no per-instance context.
- **`Scene`** (`genesis/engine/scene.py`) is the main API surface: `add_entity(...)`, then `build(n_envs=...)`, then `step()`. **`build()` compiles GPU kernels and must precede stepping** — entities can only be added before build.
- **`Simulator`** (`genesis/engine/simulator.py`) owns all active solvers and the shared state; one scene = one coupled state across all physics.

### The three orthogonal axes when adding an entity

`scene.add_entity(morph, material=..., surface=...)` composes three independent concepts:

1. **Morph** (`genesis/options/morphs.py`) — geometry + initial pose, solver-agnostic: `Box`, `Sphere`, `Plane`, `Mesh`, `URDF`, `MJCF`, `USD`. (`gs.morphs.*`)
2. **Material** (`genesis/engine/materials/`) — physical properties, and **the material picks which solver handles the entity**: `gs.materials.Rigid`, `gs.materials.MPM.Elastic`, `gs.materials.SPH.Liquid`, `gs.materials.PBD.Cloth`, etc.
3. **Surface/texture** (`genesis/options/surfaces.py`, `textures.py`) — appearance for rendering.

### Solvers and coupling

Solvers live in `genesis/engine/solvers/` — `rigid/` (the largest: `collider/`, `constraint/`, `abd/`), `mpm_solver`, `sph_solver`, `fem_solver`, `pbd_solver`, `sf_solver` (stable fluid), `kinematic_solver`, `tool_solver`. Each has matching `gs.options.*Options` (`genesis/options/solvers.py`) and entity types in `genesis/engine/entities/`.

Cross-solver interaction goes through **couplers** (`genesis/engine/couplers/`): `legacy_coupler` (explicit), `sap_coupler` (SAP), `ipc_coupler` (libuipc, optional). Changes that touch multiple solvers or coupling are high-risk — see the "When to Ask a Human" list in `AGENTS.md`.

### Quadrants kernels (`qd`) — the compute layer

Physics hot loops are **not** plain Python. Genesis compiles kernel code to CUDA/ROCm/Metal/Vulkan/CPU via **Quadrants** (`import quadrants as qd`), a Taichi fork bundled with Genesis. The patterns to recognize:

- `@qd.kernel` / `@qd.func` decorate compiled functions inside solvers (e.g. throughout `mpm_solver.py`). Code inside these obeys Quadrants' restrictions, not normal Python.
- Solver/entity **state** is held in struct-of-arrays containers defined via the metaclass system in `genesis/utils/array_class.py` (`V`/`V_VEC`/`V_MAT` wrap `qd.tensor` as Field or NDArray depending on `gs.use_ndarray`). This is how kernels read/write simulation data.
- User-facing I/O is PyTorch tensors on `gs.device`; conversions cross the torch↔quadrants boundary (zero-copy where supported).

### Parallel / batched envs

`scene.build(n_envs=N, env_spacing=(x, y))` runs N environments in lockstep on the GPU. Control APIs take batched tensors shaped `(n_envs, n_dofs)` — this batching is the basis for RL throughput. See `examples/rigid/heterogeneous_simulation.py` for non-identical envs.

## Conventions worth following

- Import idiom: `import genesis as gs`, `import genesis.utils.geom as gu`, plus `numpy as np`, `torch`.
- All config is Pydantic models under `genesis/options/` passed into `gs.Scene(...)` (`sim_options=`, `rigid_options=`, `vis_options=`, …).
- Domain errors: `gs.raise_exception("...")`; non-fatal: `gs.logger.warning(...)`.
- Tensors: always pass `device=gs.device`; precision is global (`"32"`/`"64"` via `gs.init`).
- `genesis/ext/` is vendored third-party code — excluded from formatting; avoid reformatting it.

## CLI

The package installs a `gs` console script (`genesis/_main.py`): `gs view <asset>` (visualize a Mesh/URDF/MJCF/USD), `gs play <robot>` (interactive ImGui joint control), `gs animate <images>` (compile frames to video). No subcommand prints help.

## Current branch context

Branch `so-100` adds SO-100 robot-arm grasping examples in `examples/manipulation/` (`so100_grasp_env.py`, `so100_grasp_train.py`, `so100_grasp_eval_headless.py`). The training is a two-stage RL pipeline (privileged PPO → behavior cloning) built on `rsl-rl-lib>=5.0.0`; `behavior_cloning.py` and the non-SO-100 `grasp_*.py` files are the shared scaffolding it mirrors.
