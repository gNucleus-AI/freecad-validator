# Preprocessing geometry scorer

Independent geometry scoring for `(raw, reference_clean, answer_clean)`. Runs in
FreeCAD's Python, including the FEM FreeCAD 1.1.0 Docker environment. It does not
run meshing or CalculiX. Static FEM integration uses a multiplicative adjustment:
`final_score = FEM_score * preprocessing_score`. Geometry credit is in [0, 1];
the FEM total is in [0, 100]. Correct geometry preserves the FEM score, errors
reduce it, and existing zero-score gates stay zero.

```bash
python -m pip install "gnucleus-freecad-validator[preprocess]"
python -m freecad_validator.fem.pre_process.score \
  --raw raw.step --reference reference_clean.step --candidate answer_clean.step \
  --out output/pre_process_score.json
```

```python
from freecad_validator.fem.pre_process.scorer import PreProcessScorer

result = PreProcessScorer().score_detailed(raw_path, reference_path, answer_path)
if result is not None:
    print(result.score, result.subscores)
```

## Geometry-only policy

Count original bodies **before Boolean Fragments**, with one input triple per
body. A triple where both reference and answer leave raw geometry unchanged is
skipped (`None` in Python, `{"status": "skipped", "score": null}` in the CLI).
An answer-only geometry change is an extra-edit penalty, even though the
reference left that body unchanged.

For required edits, compute continuous geometry similarity from three signals:

- `global_geometry`: volume, surface area, bounding-box dimensions, surface-type areas.
- `pointcloud_chamfer`: bidirectional surface distance in the original coordinate frame.
- `region_edit_overlap`: overlap of material changed from raw to reference versus
  raw to answer, including precision (extra edits) and recall (missing edits).

`geometry = region_edit_overlap * (0.75 * pointcloud_chamfer + 0.25 * global_geometry)`.
The body earns **1 iff geometry > 0.95**, otherwise **0** (exactly 0.95 fails).
The continuous value is retained in `details.geometry_score`. Geometry-equivalent
answers have geometry=1; an unprocessed raw answer has geometry=0 when an edit is required.

For task aggregation, let N be the number of bodies changed in the reference, C
the number of those earning 1, and E the number of answer-only changed bodies:
`score = max(0, (C - E) / N)`. Extra edits do not enlarge the denominator.
If N=0, return no score when E=0, otherwise 0. Individual extra edits have body
score=0 and `details.extra_change=true`; aggregation subtracts one point each.

The static path automatically establishes correspondence, including unchanged
bodies so extra edits can be detected. Post-Boolean fragments are not counted
separately. Matching whole-body deletions earn 1; retaining a body that should be
deleted or deleting one that should remain earns 0. A missing input file is an
error, not deletion.

## Static FEM integration

`score_step_fcstd(..., require_preprocessing=True)` automatically runs the
geometry worker against the same raw STEP, reference FCStd and candidate FCStd
used by FEM grading. No manual body mapping or task-by-task configuration is required.
The `freecad-validator fem-score --require-preprocessing` option activates this path. Non-preprocessing cases stay unchanged. Existing FEM gates
remain in force; a candidate already gated to zero does not need another geometry
evaluation.

Automatic correspondence counts raw STEP solids as original bodies and matches
saved **pre-Boolean clean inputs** one-to-one in world coordinates. The selected
analysis feature must retain input links: `PreprocessingInputs`, Boolean
Fragments `Objects`, or supported native Boolean input properties such as
MultiFuse `Shapes`. These links identify clean bodies; object labels and
visibility do not select geometry.

The preprocessing scorer never reads, compares, verifies, partitions or rebuilds
the Boolean-result geometry. Missing pre-Boolean input links are an evaluation
error, not permission to use analyzed solids. Scripts performing Boolean
operations on detached shapes must save their clean input bodies and link them
through `PreprocessingInputs`. FEM and Boolean validation remain separate.

Unmatched moved parts can be associated by unique shape descriptors; this does
not align them for scoring. Candidate matching also uses reference clean bodies.
Extra candidate bodies are penalized without increasing the denominator.

To combine with an already verified FEM report without another solver replay:

```bash
python -m freecad_validator.fem.pre_process.score --assembly \
  --raw source.step --reference reference.FCStd --candidate answer.FCStd \
  --fem-report scoring_report.json --out combined.json
```

This recomputes geometry, retains its per-body report and writes the adjusted FEM
report plus continuous reward. The caller must ensure the verified FEM report
belongs to the same inputs. For already extracted trusted data,
`score_trusted_payloads(..., preprocessing_score=geometry_score)` applies the same
policy without FreeCAD. A no-edit task with no extra edits has no geometry score
and leaves the FEM total unchanged; missing geometry evaluation is an error,
not an assumed score of 1. The report records the multiplier and original FEM
score in `subscores_details.preprocessing`; its zero weight means this is not
another term in the weighted average. The final grade is recomputed.

Topology edit counts are excluded from the score. Region overlap uses geometric
subshape matching only to locate the sampling
box; face/edge/vertex edit counts do not contribute to the score. No ICP alignment
is performed because position and orientation can be part of the requested edit.

## Input selection and limitations

- In the single-body API, STEP inputs use all solids. FCStd inputs use the unique mesh-linked geometry, or
  the unique terminal shape when there is no mesh. Ambiguous documents require
  `--raw-object`, `--reference-object`, or `--candidate-object` (object **Name**).
  The Python API uses the corresponding keyword arguments with underscores.
- Inputs are copied, not recomputed or saved in place. The solid union is compared:
  Compound wrappers, internal partitions and redundant split faces are removed.
  Thus this scorer does not validate component counts, material ownership or
  internal interfaces between touching components. Those belong to FEM setup checks.
- `--assembly` and the default static integration score corresponding original
  bodies individually. Plain `--raw` without `--assembly` is the single-body API;
  do not pass an entire assembly to that mode, which can dilute local defects.
- Non-null raw, reference and candidate paths must contain valid nonempty solid geometry.
  Explicitly missing or invalid candidate analysis geometry produces a
  `candidate_invalid` result with zero credit and a reason. Invalid trusted inputs,
  ambiguous body ownership, and kernel/infrastructure failures remain evaluation
  errors; they are not evidence that the candidate is wrong. Raw must exist; intentional whole-body deletion
  uses an explicit null clean path in the single-body Python API and is detected
  automatically in assembly mode.
- Correspondence failures identify `input_role` (`reference` or `candidate`) and
  `error_type: body_correspondence` in the worker JSON. A nonzero worker exit
  retains its reported reason in the extraction error. Such failures do not
  produce a numeric FEM or geometry score; they remain distinct from explicitly
  invalid candidate geometry.
- Both references and candidates must retain their pre-Boolean clean input
  objects. Baked results without input links cannot be evaluated by this scorer.
  The separate FEM/Boolean checks handle downstream analysis geometry.
- Failed optional splitter removal or tessellation does not discard otherwise
  valid geometry: original-body matching falls back to CAD intersections.
- The preprocessing adapter uses the FEM API's existing `timeout_seconds` limit.
  Exceeding it terminates the worker process group and raises an extraction error;
  a timeout is not treated as invalid candidate geometry.
- Region overlap defaults to 8,192 deterministic samples. Small or scattered edits
  may be undersampled; inspect the reported counts and rerun with more samples.
  Zero sampled changed volume is not proof that the shapes are equivalent.
- Reference similarity does not establish instruction compliance. Alternative
  valid patch surfaces rely on the geometric tolerances and the >0.95 threshold;
  topology differences do not receive separate penalties.

Dependencies are FreeCAD, NumPy, SciPy and `manifold3d`. The last provides
seam-independent solid equivalence checks. Install the requirements with the same
Python interpreter used to run FreeCAD. Calls use a process-global geometry cache
and should run serially within one process.

The required geometry comparison code lives in `geometry_compare/` within this
directory and is included in the package. Install the `preprocess` extra in the
Python environment used by FreeCAD, including when it differs from the host
interpreter. No experimental comparison framework or separate source tree is needed.
