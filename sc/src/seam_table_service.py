import numpy as np

np.bitwise_not is np.invert
class SeamTableService():

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
    def find_seam(cumulative_map,backtrack,r, c):
        mask = np.ones((r, c), dtype=np.bool)
        j = np.argmin(cumulative_map[-1])
        for i in reversed(range(r)):
                # Mark the pixels for deletion
            mask[i, j] = False
            j = backtrack[i, j]
        return mask

    @staticmethod
    def carve_column(img, energy_map, mode=None, object_removal_mask=None):
        # r, c = energy_map.shape
        # print('energy_map.shape {} {}'.format(r, c))
        # r, c = object_removal_mask.shape
        # print('object_removal_mask.shape {} {}'.format(r, c))
        r, c, _ = img.shape
        # print('img.shape {} {}'.format(r, c))
        M,backtrack = SeamTableService.build_minimum_seam_table(r, c, energy_map)

        mask = SeamTableService.find_seam(M,backtrack, r, c)

        mask3D = np.stack([mask] * 3, axis=2)
        img = img[mask3D].reshape((r, c - 1, 3))

        if mode == 'object_removal':
            object_removal_mask = object_removal_mask[mask].reshape((r, c - 1))
            return img, object_removal_mask

        return img


    def get_object_dimension(mask):
        rows, cols = np.where(mask > 0)
        height = np.amax(rows) - np.amin(rows) + 1
        width = np.amax(cols) - np.amin(cols) + 1
        return height, width   
