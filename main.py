import argparse
from dataset import load_image, load_mask
from model import train_model, evaluate_model
from postprocessing import postprocess
import logging

logging.basicConfig(level=logging.INFO)

def main(args):
    if args.train:
        train_model()
    if args.evaluate:
        evaluate_model()
    if args.postprocess:
        image = load_image(args.image_name)
        mask = load_mask(args.image_name)
        postprocess(image, mask)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MVA Project CLI")
    parser.add_argument('--train', action='store_true', help="Run model training")
    parser.add_argument('--evaluate', action='store_true', help="Run model evaluation")
    parser.add_argument('--postprocess', action='store_true', help="Run post-processing")
    parser.add_argument('--image_name', type=str, default="seq14_000200_0_5.png", help="Image name for post-processing")
    args = parser.parse_args()

    main(args)