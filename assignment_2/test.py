import os
from PIL import Image
import numpy as np

# Use a relative path without a leading slash
data_dir = os.path.join("data", "annotations", "trimaps")

# List all files and filter out the hidden metadata/system files
files = [f for f in os.listdir(data_dir) if not f.startswith('.')]
print(len(files))
if len(files) > 0:
    # Pick the first valid file
    target_file = files[0]
    path = os.path.join(data_dir, target_file)
    
    print(f"Attempting to open: {path}")
    
    try:
        trimap = Image.open(path)
        trimap_array = np.array(trimap)

        # Check the unique values - you'll likely see [1, 2, 3]
        print(np.unique(trimap_array))
        trimap_array[trimap_array == 1] = 255  # Background
        trimap_array[trimap_array == 2] = 0  # Unknown
        trimap_array[trimap_array == 3] = 255  # Foreground

        # Create a "viewable" version
        viewable_trimap = Image.fromarray((trimap_array).astype(np.uint8))
        viewable_trimap.save('visible_trimap.png')
    except Exception as e:
        print(f"Error opening image: {e}")
else:
    print("No valid image files found in the directory.")