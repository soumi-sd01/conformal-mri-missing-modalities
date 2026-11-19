# MRI Preprocessing Protocol

## Intensity normalization
For each patient and each available MRI modality:
1. Identify nonzero voxels.
2. Compute mean and standard deviation over nonzero voxels.
3. Apply z-score normalization to nonzero voxels.
4. Preserve background zeros.

If the nonzero standard deviation is at most `1e-8`, set the nonzero values to zero.

## Geometry
Preserves native voxel spacing and array dimensions (no resampling). Validates shape and NIfTI affine alignment.

## Labels
Raw BraTS labels are mapped as `0->0`, `1->1`, `2->2`, `4->3`. WT is labels 1/2/3, TC is labels 1/3, and ET is label 3.
