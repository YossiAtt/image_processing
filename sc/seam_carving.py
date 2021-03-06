import os
import cv2
import numpy as np
from src.cropper import Cropper


def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    # mask = os.path.join(base_dir, 'img\\mask.jpg')
    in_filename = os.path.join(base_dir, 'img\\barvaz.jpg')
    out_filename = os.path.join(base_dir, 'img\\out.jpg')
    img = cv2.imread(in_filename)
    # objectRemove =  cv2.imread(mask,0).astype(np.float64)
    saveImage = Cropper.adding_dimension(img,460,460)
    Cropper.save_result(saveImage,out_filename)


if __name__ == '__main__':
    main()
