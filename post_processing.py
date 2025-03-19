import cv2
import numpy as np
from config import OUTPUT_DIR, DIST_THRESHOLD
from utils import apply_morphological_opening, compute_distance_transform
import logging

logging.basicConfig(level=logging.INFO)

def postprocess(image_original, img_pred):
    imgray = cv2.cvtColor(img_pred, cv2.COLOR_RGB2GRAY)
    imgmorph = apply_morphological_opening(imgray)

    _, imgbin = cv2.threshold(imgmorph, 0, 255, cv2.THRESH_BINARY)
    imgdist = compute_distance_transform(imgbin)

    _, max_val, _, centre = cv2.minMaxLoc(imgdist)

    if max_val > DIST_THRESHOLD:
        cv2.imwrite(OUTPUT_DIR + 'img1_original.jpg', image_original)
        cv2.imwrite(OUTPUT_DIR + 'img2_prediction.jpg', img_pred)
        cv2.imwrite(OUTPUT_DIR + 'img3_morphopening.jpg', imgmorph)
        cv2.imwrite(OUTPUT_DIR + 'img4_DistTransformed.jpg', imgdist)

        cv2.circle(image_original, centre, int(max_val), (0, 0, 255), 2)
        cv2.circle(img_pred, centre, int(max_val), (0, 0, 255), 2)

        cv2.imwrite(OUTPUT_DIR + 'img5_predwithcircle.jpg', img_pred)
        cv2.imwrite(OUTPUT_DIR + 'img6_imgwithcircle.jpg', image_original)

        logging.info("Post-processing completed. Results saved.")
    else:
        logging.warning("Identified circle did not pass threshold requirement")