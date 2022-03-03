# /usr/bin/python3
"""Pix sfm"""

import pycolmap
from pathlib import Path
from hloc import extract_features, match_features, pairs_from_exhaustive, pairs_from_covisibility
from hloc.visualization import plot_images, read_image
from hloc.utils.viz_3d import init_figure, plot_points, plot_reconstruction, plot_camera_colmap
from pixsfm.refine_hloc import PixSfM
import matplotlib.pyplot as plt

images = Path('data/images')
input_model = Path('data/model')
outputs = Path('data/results')

sfm_pairs = outputs / 'pairs-sfm.txt'
features = outputs / 'features.h5'
matches = outputs / 'matches.h5'
sfm_dir = outputs / "sfm"
sfm_ba_dir = outputs / "sfm_ba"

feature_conf = extract_features.confs['superpoint_aachen']
matcher_conf = match_features.confs['superglue']

references = [str(p.relative_to(images))for p in images.iterdir()]

print(len(references), "mapping images")
plot_images([read_image(images / r) for r in references[:4]], dpi=50)
plt.show()

print("\n--- Extract Features ---")
extract_features.main(feature_conf, images,
                      image_list=references,
                      feature_path=features)
print("\n--- Get pairs ---")
pairs_from_covisibility.main(input_model,
                             sfm_pairs,
                             num_matched=5)
print("\n--- Match features ---")
match_features.main(matcher_conf,
                    sfm_pairs,
                    features=features,
                    matches=matches)

print("\n--- Triangulate ---")
refiner = PixSfM(conf={"dense_features": {"use_cache": True, "dtype": "half"}})
model, _ = refiner.triangulation(output_dir=sfm_dir,
                                 reference_model_path=input_model,
                                 image_dir=images,
                                 pairs_path=sfm_pairs,
                                 features_path=features,
                                 matches_path=matches
                                 )
