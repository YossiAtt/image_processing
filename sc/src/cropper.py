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
            status, img = SeamTableService.carve_column(img, energy_map)

        return img

    @staticmethod
    def remove_object(img, object_point):
        r, c, _ = img.shape

        if len(object_point) > 0:
            object_removal_mask = SeamTableService.build_mask_with_object_removal(r, c, object_point)
            indices = object_removal_mask.astype(np.uint8)  # convert to an unsigned byte
            indices *= 255
            cv2.imshow('Indices', indices)
            cv2.waitKey()     # isFinish = False
            # while not isFinish:
            #     energy_map = i_energy.calc_energy(img)
            #     energy_map_fixed = SeamTableService.energy_map_with_concern_to_object(energy_map, object_removal_mask)
            #     isFinish, img, object_removal_mask = SeamTableService.carve_column(img=img, energy_map=energy_map_fixed,
            #                                                         mode='object_removal', object_removal_mask=object_removal_mask)

        return img

    @staticmethod
    def crop_r(img, scale_r):
        img = np.rot90(img, 1, (0, 1))
        img = SeamTableService.crop_c(img, scale_r)
        img = np.rot90(img, 3, (0, 1))
        return img
