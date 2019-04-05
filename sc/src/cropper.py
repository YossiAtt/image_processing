from tqdm import trange
import numpy as np
from src.seam_table_service import SeamTableService
from src.energy_carving import Energy
import cv2
i_energy = Energy()


class Cropper():

    @staticmethod
    def crop_c(img, scale_c):
        r, c, _ = img.shape
        new_c = int(scale_c * c)
        for i in trange(c - new_c):
            energy_map = i_energy.calc_energy(img)
            img = SeamTableService.carve_column(img=img, energy_map=energy_map)

        return img
    @staticmethod
    def adding_dimension(img,out_height,out_width):
        in_height, in_width = img.shape[: 2]
        delta_row, delta_col = int(out_height - in_height), int(out_width - in_width)
        while delta_col > 0 or  delta_row > 0:
            if delta_col > 0:
                energy_map = i_energy.calc_energy(img)
                img = SeamTableService.adding_seam(img,energy_map)
                delta_col =  delta_col -1

            if delta_row > 0:
                print("rotate")
                img = np.rot90(img, 1, (0, 1))
                energy_map = i_energy.calc_energy(img)
                img = SeamTableService.adding_seam(img,energy_map)
                delta_row =  delta_row -1
                img = np.rot90(img, 3, (0, 1))

        return img


    @staticmethod
    def remove_object(img, mask):
        rotate = False
        while len(np.where(mask[:, :] > 0)[0]) > 0:

            object_height, object_width = SeamTableService.get_object_dimension(mask)
            if object_height < object_width:
                print("rotate")
                img = np.rot90(img, 1, (0, 1))
                mask = np.rot90(mask, 1, (0, 1))
                rotate = True
            else:
                print("not rotate")
            energy_map = i_energy.calc_energy(img)
            energy_map[np.where(mask[:, :] > 0)] *= -1000
            img, mask = SeamTableService.carve_column(img=img, energy_map=energy_map,
                                                                mode='object_removal', object_removal_mask=mask)
            if rotate == True:
                img = np.rot90(img, 3, (0, 1))
                mask = np.rot90(mask, 3, (0, 1))


        return img

    @staticmethod
    def crop_r(img, scale_r):
        img = np.rot90(img, 1, (0, 1))
        img = SeamTableService.crop_c(img, scale_r)
        img = np.rot90(img, 3, (0, 1))
        return img

    def save_result(out_image, filename):
        cv2.imwrite(filename, out_image.astype(np.uint8))

 