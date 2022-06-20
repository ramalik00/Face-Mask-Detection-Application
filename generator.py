# Simple script to obtain mask detector results on images which were used to present
# in the gallery of homepage 
from mask_detection import mask_detect_image,mask_detect
import cv2
import argparse

USE_GPU=False

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input", type=str, default="0")
ap.add_argument("-o", "--output", type=str, default="")
ap.add_argument("-d", "--display", type=int, default=1)
args = vars(ap.parse_args())



image=cv2.imread(args["input"])
final_image=mask_detect_image(image)

if args["output"] != "":
        cv2.imwrite(args["output"],final_image)
