#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#
import os
from argparse import ArgumentParser
import json

def run_scene(args, scene, dnum):
    data_dir = args.data_dir
    cap_max = args.cap_max

    exp_name = f"custom_{cap_max}"
    output_path = f"./eval/{exp_name}"

    if not args.skip_training:
        common_args = " --disable_viewer --eval --test_iterations -1"
        # Instant-NGP hash table arguments. log_hash_table_size can be increase to 21 or 22 if memory is not a concern
        common_args += " --minres 16 --maxres 1024 --log_hash_table_size 20 "
        # Specify number of primitives
        common_args += f" --cap_max_init {cap_max//2} --cap_max_final {cap_max} "
        # Clamping texture values leads to slightly better results but we didn't use it in the paper
        common_args += " --no_texture_clamp "

        images_dir = f"images_{dnum}"
        cmd_str = f"python train.py -s {data_dir}/{scene} -i {images_dir} -m {output_path}/{scene} {common_args}"
        print(cmd_str)
        os.system(cmd_str)

    if not args.skip_render:
        common_args = " --eval --skip_train --skip_ellipse "
        common_args += f" --no_texture_clamp "

        cmd_str = f"python render.py -s {data_dir}/{scene} -m {output_path}/{scene} {common_args}"
        print(cmd_str)
        os.system(cmd_str)

    if not args.skip_metrics:
        cmd_str = f"python metrics.py -m {output_path}/{scene}"
        print(cmd_str)
        os.system(cmd_str)

if __name__ == "__main__":
    parser = ArgumentParser(description="Custom evaluation script parameters")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=4)
    parser.add_argument("--cap_max", type=int, default=400000)
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--skip_training", action="store_true")
    parser.add_argument("--skip_render", action="store_true")
    parser.add_argument("--skip_metrics", action="store_true")
    args, _ = parser.parse_known_args()

    scenes = ["graffiti", "grocery", "table", "tripod"]
    downscale_nums = [
        4, 4, 1, 4
    ]
    for i in range(args.start, args.end):
        run_scene(args, scenes[i], downscale_nums[i])