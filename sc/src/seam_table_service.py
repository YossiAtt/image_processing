import numpy as np
import math
from random import randint,sample
from .energy_carving import Energy
from numba import jit

np.bitwise_not is np.invert
i_energy = Energy()


class SeamTableService():

    @staticmethod
    def fix_minimum_seam_table(M, backtrack, r, c, energy_map, seam_path):
        for i in range(1, r):
            col = seam_path[i-1]
            M[i-1, col] = energy_map[i-1, col]
            if col == 0:
                for j in range(col, col+2):
                    idx = np.argmin(M[i-1, j:j + 2])
                    backtrack[i, j] = idx + j
                    min_energy = M[i - 1, idx + j]
                    M[i, j] += min_energy
            elif col == 1:
                for j in range(col-1, col+2):
                    b = j if j-1 < 0 else j-1
                    idx = np.argmin(M[i-1, b:j + 2])
                    backtrack[i, j] = idx + j
                    min_energy = M[i - 1, idx + j]
                    M[i, j] += min_energy
            elif col == c:
                for j in range(col-2, col):
                    idx = np.argmin(M[i-1, j-2:j])
                    backtrack[i, j] = idx + j
                    min_energy = M[i - 1, idx + j]
                    M[i, j] += min_energy
            elif col == c-1:
                for j in range(col-2, col+1):
                    b = j if j+1 > c else j+1
                    idx = np.argmin(M[i-1, j-2:b])
                    backtrack[i, j] = idx + j
                    min_energy = M[i - 1, idx + j]
                    M[i, j] += min_energy
            else:
                for j in range(col-1, col+1):
                    idx = np.argmin(M[i - 1, j - 1: j + 1])
                    backtrack[i, j] = idx + j - 1
                    min_energy = M[i - 1, idx + j - 1]
                    M[i, j] += min_energy

        return M, backtrack

    @staticmethod
    def build_minimum_seam_table(r, c, energy_map):
        M = energy_map.copy()
        backtrack = np.zeros_like(M, dtype=np.int)

        for i in range(1, r):
            for j in range(0, c):
                # Handle the left edge of the image, to ensure we don't index -1
                if j == 0:
                    idx = np.argmin(M[i - 1, j:j + 2])
                    backtrack[i, j] = idx + j
                    min_energy = M[i - 1, idx + j]
                else:
                    idx = np.argmin(M[i - 1, j - 1:j + 2])
                    backtrack[i, j] = idx + j - 1
                    min_energy = M[i - 1, idx + j - 1]

                M[i, j] += min_energy

        return M, backtrack

    @staticmethod
    def find_seam(cumulative_map, backtrack, r, c, mode='regular'):
        seam_path = np.zeros((r,), dtype=np.uint32)
        mask = np.ones((r, c), dtype=np.bool)
        if mode == 'regular':
            j = np.argmin(cumulative_map[-1])
            for i in reversed(range(r)):
                    # Mark the pixels for deletion
                mask[i, j] = False
                j = backtrack[i, j]
                seam_path[i] = j

            return mask, seam_path
        elif mode == 'multi_path':
            multi_path = []
            j_k_minmum = np.argpartition(cumulative_map[-1],200)
            numbersList= sample(range(400), 10)
            for i in range(10):
                j = j_k_minmum[numbersList.pop()]
                for i in reversed(range(r)):
                    mask[i, j] = False
                    j = backtrack[i, j]
                    seam_path[i] = j
                multi_path.append(seam_path)
            return mask, multi_path

    @staticmethod
    def carve_column(img, energy_map, mode=None, object_removal_mask=None):
        # r, c = energy_map.shape
        # print('energy_map.shape {} {}'.format(r, c))
        # r, c = object_removal_mask.shape
        # print('object_removal_mask.shape {} {}'.format(r, c))
        r, c, _ = img.shape
        # print('img.shape {} {}'.format(r, c))
        M, backtrack = SeamTableService.build_minimum_seam_table(
            r, c, energy_map)

        mask, seam_path = SeamTableService.find_seam(M, backtrack, r, c)

        mask3D = np.stack([mask] * 3, axis=2)
        img = img[mask3D].reshape((r, c - 1, 3))

        if mode == 'object_removal':
            object_removal_mask = object_removal_mask[mask].reshape((r, c - 1))
            return img, object_removal_mask

        return img

    # @jit
    def build_image_with_extra_seam(seams_path, out_image):
        for seam in seams_path:
            r, c, _ = out_image.shape
            output = np.zeros((r, c + 1, 3))
            for row in range(r):
                col = seam[row]
                for ch in range(3):
                    if row > 2:
                        useGausian = True
                        startRow = row - 2
                        endRow = row + 3
                    if row == 0 or row > r-2:
                        useGausian = False

                    if col > 2 :
                        useGausian = useGausian & True
                        startCol = col-2
                        endCol = col+3
                    if col > c-2 or col == 0:
                        useGausian = False


                    if useGausian:
                        p = i_energy.calc_gaussian(
                            out_image[startRow:endRow, startCol: endCol, ch])
                        output[row, : col, ch] = out_image[row, : col, ch]
                        output[row, col, ch] = p
                        output[row, col + 1:, ch] = out_image[row, col:, ch]
                    elif col == 0:
                        p = np.average(out_image[row, col: col + 2, ch])
                        output[row, col, ch] = out_image[row, col, ch]
                        output[row, col + 1, ch] = p
                        output[row, col + 1:, ch] = out_image[row, col:, ch]
                    else:
                        p = np.average(out_image[row, col - 1: col + 1, ch])
                        output[row, : col, ch] = out_image[row, : col, ch]
                        output[row, col, ch] = p
                        output[row, col + 1:, ch] = out_image[row, col:, ch]
            out_image = np.copy(output)
        return out_image

    @staticmethod
    def adding_seam(img, energy_map):
        r, c = energy_map.shape
        print('energy_map.shape {} {}'.format(r, c))
        r, c, _ = img.shape
        print('img.shape {} {}'.format(r, c))
        M, backtrack = SeamTableService.build_minimum_seam_table(
            r, c, energy_map)
        mask, seams_path = SeamTableService.find_seam(
            M, backtrack, r, c, mode='multi_path')
        img = SeamTableService.build_image_with_extra_seam(seams_path, img)
        return img

    @staticmethod
    def multy_seam(img, energy_map, seams):
        seams_path = []
        masks = []
        r, c, _ = img.shape
        M, backtrack = SeamTableService.build_minimum_seam_table(
            r, c, energy_map)

        for seam in range(seams):
            mask, seam_path = SeamTableService.find_seam(
                M, backtrack, r, c)
            seams_path.append(seam_path)
            energy_map[np.where(mask[:, :] > 0)] *= 1000
            masks.append(mask)

            M, backtrack = SeamTableService.fix_minimum_seam_table(
                M, backtrack, r, c, energy_map, seam_path)

        operation_mask = np.logical_or.reduce(np.array(masks))
        operation_mask_3d = np.stack([operation_mask] * 3, axis=2)

        return operation_mask_3d, seams_path
