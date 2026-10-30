from allensdk.brain_observatory.ecephys.ecephys_project_cache import EcephysProjectCache
import os
import matplotlib.pyplot as plt


# set cache path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

MANIFEST_PATH = os.path.join(PROJECT_ROOT, 'allen_ecephys_cache', 'manifest.json')
CACHE = EcephysProjectCache.from_warehouse(manifest=MANIFEST_PATH)

# get template for natural movie 1 (movie_number=1 → natural_movie_one)
MOVIE_TEMPLATE = CACHE.get_natural_movie_template(1)

# check movie template shape (frames, height, width) and dtype
print(f"Movie template shape: {MOVIE_TEMPLATE.shape}")
print(f"Data type: {MOVIE_TEMPLATE.dtype}")

FRAME_IDX = 55
plt.imshow(MOVIE_TEMPLATE[FRAME_IDX], cmap='gray')
plt.title(f"Frame {FRAME_IDX}")
plt.axis('off')
plt.show()