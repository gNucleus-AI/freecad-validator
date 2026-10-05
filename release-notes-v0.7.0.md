# v0.7.0 — Static FEM preprocessing geometry scoring

## Highlights

This release adds geometry validation for static FEM tasks that require
preprocessing. It compares the raw source geometry, the reference's processed
geometry, and the candidate's processed geometry, then combines preprocessing
credit with the existing FEM score.

- **Original-body scoring:** evaluates edits per original body before Boolean
  Fragments, rather than treating each resulting fragment as a separate body.
- **Automatic correspondence:** recovers processed bodies from verified saved
  inputs, document history, or regrouped analysis geometry. Ambiguous ownership
  remains an evaluation error.
- **Geometry-based edit checks:** combines global geometric properties,
  point-cloud distance, and changed-material overlap. Comparisons preserve the
  original coordinate frame; no ICP alignment is applied.
- **Standalone evaluation:** adds a Python API and module CLI for preprocessing
  geometry scoring without running meshing or CalculiX.
- **Diagnostics and regression coverage:** distinguishes explicitly invalid
  candidate geometry from correspondence, kernel, and evaluation failures.

## Scoring behavior

For static FEM evaluation, `--require-preprocessing` now also enables the new
geometry scorer:

```text
final FEM score = existing FEM score × preprocessing credit
```

Preprocessing credit lies in `[0, 1]`; the FEM score lies in `[0, 100]`.

A reference-required body edit earns one credit only when its geometry similarity
is **strictly greater than 0.95**. Extra, unrequested body edits subtract credits:

```text
preprocessing credit = max(0, (correct required edits − extra edits) / required edits)
```

Bodies unchanged by both the reference and candidate are skipped. If no edits
are required and no extra edits are made, no multiplier is applied; extra edits
in that situation receive zero credit. Topology-only partition differences do
not incur a geometry penalty; existing FEM and Boolean checks remain separate.

Correct preprocessing preserves the existing FEM score. Incorrect preprocessing
reduces it, and existing zero-score gates remain zero. Reports record the
multiplier and original FEM score. Evaluation failures are not silently converted
into successful geometry scores.

The multiplier is not applied when preprocessing is not required. Existing CAD
v1/v2 scorer implementations are not changed by this release.

## Installation

Install the new `preprocess` extra in the Python environment used by FreeCAD:

```bash
python -m pip install "gnucleus-freecad-validator[preprocess]==0.7.0"
```

This extra adds the pinned dependency `manifold3d==3.5.4`. FreeCAD remains an
external runtime dependency.

Example standalone assembly evaluation (run with FreeCAD's Python, or a
compatible interpreter whose `PYTHONPATH` includes FreeCAD's binding directory):

```bash
python -m freecad_validator.fem.pre_process.score --assembly \
  --raw source.step \
  --reference reference.FCStd \
  --candidate answer.FCStd \
  --out preprocessing.json
```

## Known limitations

- A detached fused snapshot can still be mistaken for a single original body,
  producing incorrect geometry credit. This remains a known limitation of this
  release.
- Region-overlap evaluation uses deterministic sampling; small or scattered
  edits may be undersampled.
- Reference similarity alone does not establish instruction compliance or
  validate material ownership and internal interfaces.

## Changes

- [PR #35: Add preprocessing geometry scoring to static FEM validation](https://github.com/gNucleus-AI/freecad-validator/pull/35)
- [Compare v0.6.2…v0.7.0](https://github.com/gNucleus-AI/freecad-validator/compare/v0.6.2...v0.7.0)
