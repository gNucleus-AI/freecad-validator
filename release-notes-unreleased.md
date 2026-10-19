# Unreleased

## FEM scoring changes

A trusted `mesh_budget.json` beside the reference can configure a `local_max`
node cap for one body. Body correspondence also runs for plain FEM cases that
enable this budget. A missing target or local overflow gives zero mesh-budget
credit. In plain cases, candidate body-correspondence errors also give zero
mesh-budget credit while other scoring continues. Reference correspondence
errors and preprocessing-task error handling remain unchanged.

The default full-credit tolerance for maximum von Mises stress and maximum shear
stress increases from 8% to 10%. This applies to all FEM cases using the default
stress tolerance, including cases without a `mesh_budget.json`. Displacement
tolerance remains 5%.

Rescoring an existing submission can therefore produce different stress scores
and overall scores. Explicitly supplied stress tolerances remain unchanged.

Local mesh-budget configuration errors now identify the configured target body
when it is missing from the reference body correspondence.
