# Autoflatten Examples

This directory contains example scripts demonstrating advanced usage of autoflatten.

## Examples

### `custom_patch_example.py`

Demonstrates the flexible patch configuration capabilities introduced to support arbitrary regions beyond the standard 5-cut template.

**Key features demonstrated:**
- Creating patches with custom cut names (e.g., `occipital_cut`, `ventral_cut`)
- Using arbitrary numbers of cuts (not limited to 5)
- Creating isolated patches without a medial wall
- Creating custom template JSON files
- Validating patch topology with `validate_patch_topology()`

**Run the example:**
```bash
python examples/custom_patch_example.py
```

This is particularly useful for:
- Creating patches for specific brain regions (e.g., just occipital pole or temporal lobe)
- Defining custom anatomical regions of interest
- Creating reusable templates for non-standard patches

## Creating Your Own Custom Patches

To create a custom patch template:

1. **Define vertices on a template surface (e.g., fsaverage):**
   ```python
   import numpy as np
   from autoflatten.utils import save_json
   
   # Define your patch boundaries
   template_dict = {
       "lh_mwall": [vertex_ids...],  # Optional medial wall
       "lh_custom_cut1": [vertex_ids...],  # Your custom cuts
       "lh_custom_cut2": [vertex_ids...],
       "rh_mwall": [vertex_ids...],
       "rh_custom_cut1": [vertex_ids...],
       "rh_custom_cut2": [vertex_ids...],
   }
   
   # Save to JSON
   save_json("my_custom_template.json", template_dict)
   ```

2. **Use the custom template with autoflatten:**
   ```bash
   autoflatten /path/to/subject --template-file my_custom_template.json
   ```

3. **Validate the topology (optional but recommended):**
   ```python
   from autoflatten import validate_patch_topology
   
   is_valid, issues, info = validate_patch_topology(
       vertex_dict, subject="your_subject", hemi="lh"
   )
   ```

## Notes

- The `mwall` key is reserved for medial wall vertices (but can be empty)
- All other keys are treated as cuts and processed dynamically
- Cut names can be arbitrary strings (no hardcoded expectations)
- Patches should have disk topology (single connected component, one boundary loop) for successful flattening
