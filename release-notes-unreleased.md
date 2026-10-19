# Unreleased

## FEM scoring changes

The default full-credit tolerance for maximum von Mises stress and maximum shear
stress increases from 8% to 10%. This applies to all FEM cases using the default
stress tolerance, including cases without a `mesh_budget.json`. Displacement
tolerance remains 5%.

Rescoring an existing submission can therefore produce different stress scores
and overall scores. Explicitly supplied stress tolerances remain unchanged.

Local mesh-budget configuration errors now identify the configured target body
when it is missing from the reference body correspondence.
