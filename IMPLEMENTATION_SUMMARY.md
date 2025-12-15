# Summary: Flexible Template Logic for Arbitrary Patches

## Issue
The original issue requested extending the template logic to:
1. Allow arbitrary patches/regions beyond the standard 5-cut template
2. Support flexible cut configurations (e.g., occipital pole only, temporal pole only)
3. Enable users to create custom patches on fsaverage that can be applied to any subject
4. Not hardcode expected cuts or medial wall
5. Add topology checks on patches

## Implementation

### Changes Made

#### 1. Core Module (`autoflatten/core.py`)
- **Dynamic cut extraction**: Replaced hardcoded `cut_names = ["calcarine", "medial1", "medial2", "medial3", "temporal"]` with dynamic extraction from `vertex_dict.keys()`
- **New function**: Added `validate_patch_topology()` to check:
  - Single connected component
  - Disk topology (one boundary loop)
  - Reasonable patch size
- **Updated docstrings**: Documented support for arbitrary cut names

#### 2. Template Module (`autoflatten/template.py`)
- **Flexible merging**: Made `merge_small_components()` accept optional `max_cuts` parameter (default None = keep all)
- **Generic classification**: Updated `classify_cuts_anatomically()` to fall back to generic names (cut1, cut2, etc.) when not applicable
- **Updated `identify_surface_components()`**: Added `max_cuts` and `classify_anatomically` parameters for flexibility

#### 3. Tests
- Updated `test_merge_small_components()` to test both old (max_cuts=5) and new (max_cuts=None) behavior
- Added `test_validate_patch_topology()` to test topology validation
- All 151 tests pass

#### 4. Documentation & Examples
- Created `examples/custom_patch_example.py` demonstrating:
  - Custom cut names
  - Arbitrary number of cuts
  - Isolated patches without medial wall
  - Template creation
  - Topology validation
- Added `examples/README.md` with comprehensive usage guide
- Updated main `README.md` with flexible patch section

#### 5. API Exports
- Updated `autoflatten/__init__.py` to export key functions for programmatic use:
  - `ensure_continuous_cuts`
  - `fill_holes_in_patch`
  - `map_cuts_to_subject`
  - `refine_cuts_with_geodesic`
  - `validate_patch_topology`
  - `identify_surface_components`

## Key Design Decisions

### 1. Convention: 'mwall' is Reserved
- The `mwall` key is reserved for medial wall vertices (can be empty)
- All other keys are treated as cuts and processed dynamically
- This provides a clear, simple convention for users

### 2. Backward Compatibility
- Default behavior for standard 5-cut template unchanged
- `max_cuts=None` enables new flexible behavior
- `classify_anatomically=True` maintains anatomical naming when appropriate

### 3. Topology Validation
- New optional validation step to catch topology issues early
- Provides clear feedback on what's wrong (disconnected components, multiple loops, etc.)
- Users can validate before attempting potentially slow flattening operations

## Usage Examples

### Example 1: Custom Patch with Arbitrary Cuts
```python
from autoflatten.utils import save_json

# Create a template with custom regions
template_dict = {
    "lh_mwall": [100, 101, 102],
    "lh_occipital_boundary": [1, 2, 3, 4, 5],
    "lh_ventral_boundary": [10, 11, 12, 13],
    "rh_mwall": [100, 101, 102],
    "rh_occipital_boundary": [1, 2, 3, 4, 5],
    "rh_ventral_boundary": [10, 11, 12, 13],
}

save_json("occipital_patch_template.json", template_dict)
```

Then use: `autoflatten subject --template-file occipital_patch_template.json`

### Example 2: Validate Before Flattening
```python
from autoflatten import validate_patch_topology

is_valid, issues, info = validate_patch_topology(
    vertex_dict, subject="sub-01", hemi="lh"
)

if not is_valid:
    print("Topology issues:", issues)
    print("Info:", info)
```

## Testing

All tests pass (151 passed, 6 skipped):
```bash
cd /home/runner/work/autoflatten/autoflatten
python -m pytest autoflatten/tests/ -v
```

Specific test coverage:
- `test_merge_small_components`: Tests flexible merging with and without max_cuts
- `test_validate_patch_topology`: Tests topology validation
- `test_classify_cuts_anatomically`: Tests fallback to generic naming
- All existing tests: Ensure backward compatibility

## Benefits

1. **Flexibility**: Users can create patches for any brain region
2. **Simplicity**: No need to conform to specific cut names or counts
3. **Validation**: Can catch topology issues before expensive flattening
4. **Reusability**: Custom templates can be applied across subjects
5. **Backward Compatible**: Existing workflows continue to work unchanged

## Future Enhancements

Potential future improvements:
1. Add CLI command to create template from manual vertex selection
2. Add visualization of custom patches before flattening
3. Support for multiple disconnected patches in a single template
4. Automatic topology fixing for common issues

## Files Modified

- `autoflatten/core.py`: Dynamic cut handling, topology validation
- `autoflatten/template.py`: Flexible merging and classification
- `autoflatten/__init__.py`: Export key functions
- `autoflatten/tests/test_core.py`: Added topology validation test
- `autoflatten/tests/test_template.py`: Updated merging test
- `README.md`: Documented flexible patches
- `examples/custom_patch_example.py`: Comprehensive examples (new)
- `examples/README.md`: Usage guide (new)

## Commits

1. `255800b`: Make template logic flexible for arbitrary patches
2. `03f895f`: Add topology validation and examples for flexible patches
3. `9737251`: Update README with flexible patch documentation
