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
    def build_mask_with_object_removal(r, c, object_point):
        mask = np.zeros((r, c), dtype=np.bool)
        for point in object_point:
            # x = point[0] , y = point[1]
            mask[point[0], point[1]] = True
        return mask

    @staticmethod
    def energy_map_with_concern_to_object(energy_map, mask):
        energy_map[mask] = -10000
        return energy_map

    @staticmethod
    def carve_column(img, energy_map, mode=None, object_removal_mask=None):
        r, c, _ = img.shape
        M, backtrack = SeamTableService.build_minimum_seam_table(r, c, energy_map)

        # Find the position of the smallest element in the
        # last row of M
        j = np.argmin(M[-1])

        if mode == 'object_removal':
            print("the minumum path is :{}".format(M[-1, j]))
            if M[-1, j] > 0:
                return True, img ,object_removal_mask

        # Create a (r, c) matrix filled with the value True
        # We'll be removing all pixels from the image which
        # have False later
        mask = np.ones((r, c), dtype=np.bool)

        for i in reversed(range(r)):
            # Mark the pixels for deletion
            mask[i, j] = False
            j = backtrack[i, j]
        # Since the image has 3 channels, we convert our
        # mask to 3D
        if mode == 'object_removal':
            object_removal_mask = object_removal_mask[mask].reshape((r, c - 1))

        mask = np.stack([mask] * 3, axis=2)

        # Delete all the pixels marked False in the mask,
        # and resize it to the new image dimensions
        img = img[mask].reshape((r, c - 1, 3))


        return False, img, object_removal_mask


def remove_redundent_point(minmum_path_indexes, object_point):
    if len(minmum_path_indexes) > 0:
        minmum_path_indexes_dict = dict(minmum_path_indexes)
        print("len of removal objects is :{}".format(len(object_point)))
        for point in object_point:
            point_minumum_index = minmum_path_indexes_dict.get(point[0])
            if (point_minumum_index is not None) and (point_minumum_index == point[1]):
                object_point.remove(point)
        print("len of removal objects is: {}".format(len(object_point)))
    return object_point
