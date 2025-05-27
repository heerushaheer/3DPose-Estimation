#!/usr/bin/env python
# Copyright (c) OpenMMLab. All rights reserved.

import argparse
import json
import torch
import numpy as np

from mmengine.config import Config
from mmengine.registry import init_default_scope
from mmpose.registry import MODELS
from mmpose.apis import inference_pose_lifter_model


def parse_args():
    parser = argparse.ArgumentParser(
        description='Demo script for lifting 2D keypoints to 3D using a PoseLifter.'
    )
    parser.add_argument(
        '--json-file',
        type=str,
        required=True,
        help='Path to the input JSON file containing 2D keypoints')
    parser.add_argument(
        '--out-json',
        type=str,
        required=True,
        help='Path to save the output 3D keypoints JSON')
    parser.add_argument(
        'pose_lifter_config',
        type=str,
        help='Config file for Pose Lifter model'
    )
    parser.add_argument(
        'pose_lifter_checkpoint',
        type=str,
        help='Checkpoint file for Pose Lifter model'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cpu',
        help='Device to run inference (e.g. "cuda:0" or "cpu")'
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # 1. Initialize default scope for MMPose
    init_default_scope('mmpose')

    # 2. Load 2D keypoints JSON
    with open(args.json_file, 'r') as f:
        data_2d = json.load(f)  # Expected: { "instances": [ { "keypoints": [...] }, ... ] }

    if 'instances' not in data_2d:
        raise ValueError("Invalid JSON format. Expected a top-level 'instances' key.")

    # 3. Load config and build Pose Lifter model
    cfg = Config.fromfile(args.pose_lifter_config)

    model = MODELS.build(cfg.model)

    # 4. Load checkpoint
    checkpoint = torch.load(args.pose_lifter_checkpoint, map_location=args.device)
    if 'state_dict' in checkpoint:
        checkpoint = checkpoint['state_dict']  # Sometimes the weights are under 'state_dict'
    model.load_state_dict(checkpoint, strict=False)
    model.to(args.device)
    model.eval()

    # 5. Parse 2D keypoints into a format recognized by `inference_pose_lifter_model`
    #    Typically a list of dicts: each with 'keypoints' (N x 3).
    #    If your data has (x,y) only => add dummy confidence=1.0.
    #    Example structure to feed into `inference_pose_lifter_model`:
    #    results_2d = [
    #       { 'keypoints': np.array([[x1,y1,score1], ..., [xN,yN,scoreN]], dtype=np.float32) },
    #       ...
    #    ]

    results_2d = []
    for i, inst in enumerate(data_2d['instances']):
        if 'keypoints' not in inst:
            raise ValueError(f"Instance {i} missing 'keypoints' field.")

        kp_list = inst['keypoints']  # e.g. 34 values if 17 joints have (x,y)

        # Detect if we have only (x,y) or (x,y,score).
        # If we have 17 joints and NO confidence => length == 34 => add dummy conf.
        # If we have 17 joints and confidence => length == 51 => already correct multiple of 3.
        if len(kp_list) % 2 == 0:
            # (x,y) only => add dummy confidence=1.0
            new_list = []
            for idx in range(0, len(kp_list), 2):
                x_ = kp_list[idx]
                y_ = kp_list[idx + 1]
                new_list.extend([x_, y_, 1.0])  # appended dummy conf=1.0
            kp_list = new_list
        else:
            # Already (x,y,score) presumably. 
            # If the length is not multiple of 3, it's invalid.
            if len(kp_list) % 3 != 0:
                raise ValueError(
                    f"Instance {i} keypoints length {len(kp_list)} not multiple of 3!"
                )

        kp_array = np.array(kp_list, dtype=np.float32).reshape(-1, 3)
        results_2d.append({'keypoints': kp_array})

    # 6. Run inference to get 3D keypoints
    results_3d = inference_pose_lifter_model(model, results_2d)

    # 7. Save to output JSON
    with open(args.out_json, 'w') as f:
        json.dump(results_3d, f, indent=2)

    print(f"3D keypoints saved to {args.out_json}")


if __name__ == '__main__':
    main()
