# Preprocessing geometry scoring

Compare raw STEP geometry with the saved clean geometry in the reference and
candidate. Enable it through `FEMValidator(require_preprocessing=True)` or
`freecad-validator fem-score ... --require-preprocessing`.

## Score

For each original body or required addition:

```text
geometry = region_overlap * (0.75 * surface_similarity + 0.25 * global_similarity)
body_credit = 1 if geometry > 0.95, otherwise 0
G = max(0, (correct_required_edits - extra_edits) / required_edits)
final_score = gated_FEM_score * G
```

Whole-body deletions are compared explicitly. Unchanged bodies are skipped unless
the candidate makes an extra edit. Required additions join the denominator; extra
candidate bodies do not. If no edits are required and no extra edits occur, G is
not applicable. This is distinct from missing evaluation evidence.

Geometry equality and edits use world-space physical measures, surface distances
and sampled occupied-volume overlap. No ICP alignment or topology-count penalty
is used for geometry credit. Internal interfaces and material assignments remain
part of the FEM setup checks. If the surface/global upper bound cannot exceed
0.95, the body receives zero without region sampling; the report records the bound,
`geometry_score: null` and `region_evaluated: false`.

## Saved inputs and read-only evaluation

Save independent clean bodies after the requested edits and before assembly
fusion or Boolean Fragments. Preserve links such as `PreprocessingInputs`,
Boolean `Objects` or `Shapes`. Hidden clean bodies are supported. Verified detached
history may also be used, but raw imports or final fused fragments alone cannot
replace independent clean bodies. A single prepared solid used directly for
meshing is sufficient for a single-original-body task.

Saved inputs are checked against the mesh-linked analysis geometry using sampled
union IoU >= 0.95. Correspondence uses sampled overlaps and one-to-one descriptor
matching for moved bodies; scoring retains their actual positions. Disjoint saved
pieces of one original are compared as one logical occupied union.

The scorer never cuts, fuses, heals, refines, reconstructs, remeshes or recomputes
evaluated geometry. Shape readers retain saved shapes without deep copies and
preserve world placement after the source document closes. Samples are finite:
small local differences can be missed, and sampled IoU is not an exact proof of
geometric equivalence.

Missing usable clean inputs in either document return `missing_clean_bodies`
with score 0, input role and reason. Invalid candidate geometry returns
`candidate_invalid` with score 0. Invalid trusted geometry, ambiguous
correspondence, failed queries and runtime failures are evaluation errors, not
measured candidate scores. The FEM report retains the full geometry evaluation
under `subscores_details.preprocessing`, including per-body results and the
original FEM score. The multiplier is applied once.

## Runtime dependencies

The runtime needs FreeCAD, NumPy, SciPy and OCP. Install the package's `preprocess`
extra in a compatible pip environment; it supplies `cadquery-ocp==7.9.3.1.1`.
For conda FreeCAD, use the compatible conda `ocp` package (validated with
`ocp=7.9.3.1`) rather than overlaying pip's different VTK dependencies.

Large point-in-solid batches run in an isolated OCP worker. Interpreter discovery
checks the current Python, FreeCAD's environment and Python executables on PATH,
probing actual worker imports once per process. Failures remain evaluation errors;
there is no fallback that changes the scoring method. The regular FEM adapter
timeout also covers preprocessing and its child processes.

The single-body Python API supports explicit null clean paths for deletion and
`raw=None` with a reference for additions. Non-null paths must exist and contain
valid solid geometry. Assemblies should use the normal FEM integration or the
existing `--assembly` geometry CLI rather than being treated as one large body.
