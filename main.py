# /usr/bin/python3
"""Pix sfm"""

from pathlib import Path
from hloc import extract_features, match_features, pairs_from_covisibility
from pixsfm.refine_hloc import PixSfM

images = Path('data/images')
input_model = Path('data/input_model')
outputs = Path('data/results')

sfm_pairs = outputs / 'pairs-sfm.txt'
features = outputs / 'features.h5'
matches = outputs / 'matches.h5'
sfm_dir = outputs / "sfm"

feature_conf = extract_features.confs['superpoint_aachen']
matcher_conf = match_features.confs['superglue']

references = [str(p.relative_to(images))for p in images.iterdir()]

extract_features.main(feature_conf, images,
                      image_list=references,
                      feature_path=features)
pairs_from_covisibility.main(input_model,
                             sfm_pairs,
                             num_matched=5)
match_features.main(matcher_conf,
                    sfm_pairs,
                    features=features,
                    matches=matches)

refiner = PixSfM(conf="/dependencies/pixel-perfect-sfm/pixsfm/configs/low_memory.yaml")
model, _ = refiner.triangulation(output_dir=sfm_dir,
                                 reference_model_path=input_model,
                                 image_dir=images,
                                 pairs_path=sfm_pairs,
                                 features_path=features,
                                 matches_path=matches
                                 )
