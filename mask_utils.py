import os
import glob

import numpy as np
from loguru import logger as guru
from sam2.build_sam import build_sam2_video_predictor


def init_sam_model(checkpoint_dir: str, model_cfg: str, device):
    predictor = build_sam2_video_predictor(model_cfg, checkpoint_dir)

    guru.info(f"loaded model checkpoint {checkpoint_dir}")
    return predictor




