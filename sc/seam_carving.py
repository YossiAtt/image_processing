
from imageio import imread, imwrite
from src.cropper import  Cropper
import os
from src.seam_git import SeamCarver
def main():
    base_dir =  os.path.dirname(os.path.abspath(__file__))
    mask = os.path.join(base_dir, 'img\\mask.jpg')
    in_filename = os.path.join(base_dir, 'img\\barvaz.jpg')
    out_filename  = os.path.join(base_dir, 'img\\barvaz1.jpg')
    img = imread(in_filename)
    r, c = img.shape[: 2]
    seam = SeamCarver(in_filename,out_width=c,out_height=r,object_mask=mask)
    seam.object_removal()
    seam.save_result(out_filename)

if __name__ == '__main__':
    main()