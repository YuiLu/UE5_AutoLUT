import numpy as np
from PIL import Image
from inference import Inference
import argparse

def inference(args):
    grader = Inference(config=args.config)
    video_path = args.video_path
    lut_path = args.lut_path
    save_path = args.save_path

    seed = args.seed
    steps = args.steps
    size = args.size

    output_path = grader(video_path, lut_path, save_path, seed, steps, size)
    print(f"Result saved at {output_path}")

def parse_arguments():
    parser = argparse.ArgumentParser(description="Animate images using given parameters.")
    parser.add_argument('--video_path', type=str, required=True, help='Path to the input videos.')
    parser.add_argument('--lut_path', type=str, required=True, help='Path to the lut file.')
    parser.add_argument('--save_path', type=str, required=True, help='Path to save evaluation results.')
    parser.add_argument('--config', type=str, default='configs/prompts/video_demo.yaml', help='Path to the configuration file.')
    parser.add_argument('--seed', type=int, help='Seed value.', default=42)
    parser.add_argument('--steps', type=int, help='Number of steps for the animation.', default=25)
    parser.add_argument('--size', type=int, help='Size of the image.', default=512)
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_arguments()
    inference(args)

