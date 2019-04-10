import numpy as np
from scipy.ndimage.filters import convolve
from scipy.signal import convolve2d
class Energy():
    def __init__(self):
        filter_du = np.array([
            [1.0, 2.0, 1.0],
            [0.0, 0.0, 0.0],
            [-1.0, -2.0, -1.0],
        ])
        # This converts it from a 2D filter to a 3D filter, replicating the same
        # filter for each channel: R, G, B
        self.filter_du = np.stack([filter_du] * 3, axis=2)

        filter_dv = np.array([
            [1.0, 0.0, -1.0],
            [2.0, 0.0, -2.0],
            [1.0, 0.0, -1.0],
        ])

        self.gaussian_kernel = np.array([[1 / 256, 4  / 256,  6 / 256,  4 / 256, 1 / 256],
                                        [4 / 256, 16 / 256, 24 / 256, 16 / 256, 4 / 256],
                                        [6 / 256, 24 / 256, 36 / 256, 24 / 256, 6 / 256],
                                        [4 / 256, 16 / 256, 24 / 256, 16 / 256, 4 / 256],
                                        [1 / 256, 4  / 256,  6 / 256,  4 / 256, 1 / 256]])
        # This converts it from a 2D filter to a 3D filter, replicating the same
        # filter for each channel: R, G, B
        self.filter_dv = np.stack([filter_dv] * 3, axis=2)
        # self.filter_gaussian = np.stack([gaussian_kernel] * 3, axis=2)
    def calc_energy(self,img):
        img = img.astype('float32')
        convolved = np.absolute(convolve(img, self.filter_du)) + np.absolute(convolve(img, self.filter_dv))
        # We sum the energies in the red, green, and blue channels
        energy_map = convolved.sum(axis=2)

        return energy_map

    def calc_gaussian(self,img):
        img = img.astype('float32')
        convolved = convolve2d(img, self.gaussian_kernel,mode="valid")
        return convolved[0][0]

