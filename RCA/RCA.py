import numpy as np
import cv2
import math
import os
import matplotlib.pyplot as plt
from PIL import Image
from scipy.sparse import lil_matrix as sparse
from scipy.special import lambertw


class RetinalCompression:
    def __init__(self):
        self.a = .98  # The weighting of the first term in the original equation
        self.r2 = 1.05  # The eccentricity at which density is reduced by a factor of four (and spacing is doubled)
        self.re = 22  # Scale factor of the exponential. Not used in our version.
        self.dg = 33162  # Cell density at r = 0
        self.C = (3.4820e+04 + .1)  # Constant added to the integral to make sure that if x == 0, y == 0.
        self.inS, self.inR, self.inC, self.inD = int, int, int, int  # Dimensions of the selected image (pixel space)
        self.bg_color = 127  # Background color for the generated image.
        self.W = None  # Placeholder for spare matrix used for image transformation when using series_dist method
        self.msk = None  # Placeholder for mask when using series_dist method

    def fi(self, r):
        # Integrated Ganglion Density formula (3), taken from Watson (2014). Maps from degrees of visual angle to the
        # amount of cells.
        return self.C - np.divide((np.multiply(self.dg, self.r2 ** 2)), (r + self.r2))

    def fii(self, r):
        # Inverted integrated Ganglion Density formula (3), taken from Watson (2014).
        return np.divide(np.multiply(self.dg, self.r2 ** 2), (self.C - r)) - self.r2

    @staticmethod
    def cones(r):
        return 200 * np.exp(-0.75 * r) + 11.5

    @staticmethod
    def cones_i(r):
        return 11.5 * r - 266.666666666667 * np.exp(-0.75 * r) + 266.666666666667

    @staticmethod
    def cones_ii(r):
        return (2*r)/23 + (4*lambertw((400*np.exp(400/23 - (3*r)/46))/23, k=0))/3 - 1600/69

    @staticmethod
    def load_image(*im):
        # Method that opens a dialogue window to select an image. The dimensions for the image are stored as class
        # fields so that they can be used later on.
        if im:
            file = im[0]
        else:
            print('No file')
        image = cv2.imread(file, 3)
        [r, c, d] = image.shape  # Determine dimensions of the selected image (pixel space)
        dif = r - c  # Determine the difference between rows and columns. Used for zero-padding of non-
        s = dif / 2
        if dif != 0:
            mval = max(r, c)
            im2 = np.zeros((mval, mval, d), dtype=np.uint8)
            if dif < 0:
                im2[int(abs(s)):int((mval - abs(s))), :, :] = image
            if dif > 0:
                im2[:, int(abs(s)):int((mval - s)), :] = image
            image = im2
            print(np.max(image))
            print(np.min(image))
        print(image.shape)
        return image

        #if dif < 0:
        #    # Zero padding in y-direction (if needed) to make the image a squared image.
        #    image = np.pad(image, [-dif / 2, 0], mode='constant')
        #elif dif > 0:
        #    # Zero padding in x-direction (if needed) to make the image a squared image.
        #    image = np.pad(image, [0, dif / 2], mode='constant')


    @staticmethod
    def show_image(img):
        # Parameters: Im = array_like
        #                   Numpy array containing an image
        # A static method for displaying any type of image.
        dim = (512, 512)
        resized = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
        cv2.imshow('image', resized)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    @staticmethod
    def mask(im, mask, average=0):
        mask = np.reshape(mask, [im.shape[0], im.shape[1]])
        if im.shape[2] == 3:
            mask = np.repeat(mask[:, :, np.newaxis], 3, axis=2)
        if average == 0:
            im2 = np.multiply(im, mask)
        if average == 1:
            im2 = np.multiply(im, mask)
        return im2

    def distort_image(self, image, fov=20, out_size=256, decomp=0, type=1, series=0):
        # Parameters:
        #     image = array_like
        #               Numpy array containing an image
        #     fov = integer
        #               Field of view coverage. Diameters in visual angle. When
        #               decompressing an image, it is advised that the value is set to the fov of the original image.
        #               range[1, 100]
        #     out_size = integer
        #               Determines the size of the output image in pixels. The value is
        #               the size of the output image on one axis. Output image is always a square image. When
        #               decompressing an image, it is advised that the value is set to the size of the original image.
        #     decomp = integer
        #               Set to 1 to get a decompression of a distorted input image. Set to 0 to have compression.
        #               range[0, 1]
        #     type = integer
        #               Set to 0 for photoreceptor (cones) based distortion, set to 1 for Ganglion cell based
        #               distortion
        #     series = integer
        #               Set to 1 to store the sparse transformation matrix which can be used for a series of
        #               transformations.
        # Notes:
        # Main method for image distortion. This can be used for both forward compression (normal image to Ganglion
        # compressed image, type_d = 0), or decompressing (Ganglion/Photoreceptor compressed to normal image, type_
        # d = 1) an image. Decompression is intended to be used to create a normalized visualization of a
        # Ganglion/Photocell-compressed images. These normalized representations can be used as an indication of
        # information loss. As the original image is in pixel-space, and the model assumes an input that is described in
        # visual angle eccentricity, there is a need to describe the image coverage on the visual field in degrees of
        # visual angle. Therefore, the model needs the user to input how much of the field of view is covered by the
        # image (fov) in degrees of visual angle. The range should be between 1 and 100 degrees of visual angle.
        # The method works on the radial distance of each pixel which will be remapped according to the amount of cells
        # involved in processing the image. A constant distance between the cells is assumed.For remapping, we use an
        # inverse mapping approach in which each pixel in the new image is determined by taking it from the original
        # image. This method requires a inverted integrated cell density function.

        if self.W is None:

            try:
                [self.inR, self.inC, self.inD] = image.shape  # Determine dimensions of the selected image (pixel space)
            except:
                #image = np.repeat(image[:, :, np.newaxis], 3, axis=2)
                [self.inR, self.inC] = image.shape  # Determine dimensions of the selected image (pixel space)
                self.inD = None
            self.inS = max(self.inR, self.inC)  # Determine the largest dimension of the image

            # Parameter e represents the radius of visual field coverage (visual angle).
            e = fov / 2

            # Determine the radius of the in-and output image.
            in_radius = self.inS / 2
            out_radius = out_size / 2

            # Calculate the number of Ganglion/Photo cells covering the radius of the image, given the field of view
            # coverage of the input image. We run the degrees of visual angle (eccentricity) through the integrated
            # Ganglion/Photo cell density function which gives us the total amount of cells involved with processing the
            # image along the radius given the covered visual field.
            if type == 0:
                n_cells = self.cones_i(e)
            if type == 1:
                n_cells = self.fi(e)

            # Decomp determines decompression. If set to 1, it is assumed that the image is compressed already, and
            # should be normalized. If set to 0, it is assumed the image needs to be distorted.
            if decomp == 0:
                # How many degrees are modeled within a pixel of the image? This can be determined by dividing the
                # visual angle of eccentricity by half the image dimension. This will be used to adjust the new radial
                # distances for each pixel (expressed in degrees) to pixel distances.
                deg_per_pix = e / in_radius
                # The n_cells variable represents the amount of cells along the radius of the covered field of view
                # (visual angle eccentricity). Therefore, the total amount of cell involved along the diameter of the
                # image will be from -n_cells to n_cells. The image pixels are expressed in number of total cells
                # involved in processing the image up to each individual pixel.
                t = np.linspace(-n_cells, n_cells, num=out_size)
            elif decomp == 1:
                # When going from distorted image to normalized image, we have to take the inverse of the inverse, thus
                # the regular integrated function. The new radial distances for each pixel will thus be given in the
                # number of retinal cells. Therefore, this has to be converted to number of pixels. This is done by
                # calculating the number of cells per pixels.
                cell_per_pix = n_cells / in_radius
                # The image is expressed in visual angles.
                t = np.linspace(-e, e, num=out_size)

            x, y = np.meshgrid(t, t)
            x = np.reshape(x, out_size ** 2)
            y = np.reshape(y, out_size ** 2)

            # For every pixel, calculate its angle, and radius.
            ang = np.angle(x + y * 1j)
            rad = np.abs(x + y * 1j)

            if decomp == 0:
                # Calculate a mask that covers all pixel beyond the modeled fov coverage, for better visualization
                # (optional)
                msk = (rad <= n_cells)
                # Calculate the new location of each pixel. Inverse mapping: For each pixel in the new image, calculate
                # from which pixel in the original image it gets it values.
                if type == 0:
                    new_r = self.cones_ii(rad) / deg_per_pix
                if type == 1:
                    new_r = self.fii(rad) / deg_per_pix
                # Use angle and new radial values to determine the x and y coordinates.
                x_n = np.multiply(np.cos(ang), new_r) + self.inS / 2
                y_n = np.multiply(np.sin(ang), new_r) + self.inS / 2
            elif decomp == 1:
                # Calculate a mask that covers all pixel beyond the modeled fov coverage, for better visualization
                # (optional)
                msk = (rad <= fov)
                # Calculate the new location of each pixel. Inverse mapping: For each pixel in the new image, calculate
                # from which pixel in the original image it gets it values.
                if type == 0:
                    new_r = self.cones_i(rad) / cell_per_pix
                if type == 1:
                    new_r = self.fi(rad) / cell_per_pix
                # Use angle and new radial values to determine the x and y coordinates.
                x_n = np.multiply(np.cos(ang), new_r) + in_radius
                y_n = np.multiply(np.sin(ang), new_r) + in_radius

            # The method used for image conversion. A sparse matrix that maps every pixel in the old image, to each
            # pixel in the new image via inverse mapping, is used.
            # Build a spare matrix for image conversion.
            W = sparse((out_size ** 2, self.inS ** 2), dtype=np.float)
            # Sometimes division by 0 might happen. This line makes sure the user won't see a warning when this happens.
            np.seterr(divide='ignore', invalid='ignore')
            for i in range(out_size ** 2):
                # Pixel indices will almost always not be a perfect integer value. Therefore, the value of the new pixel
                # is value is determined by taking the average of all pixels involved. E.g. a value of 4.3 is converted
                # to the indices 4, and 5. The RGB values are weighted accordingly (0.7 for index 4, and 0.3 for index
                # 5). Additionally, boundary checking is used. Values can never be smaller than 0, or larger than the
                # maximum index of the image.
                x = np.minimum(np.maximum([math.floor(y_n[i]), math.ceil(y_n[i])], 0), self.inS - 1)
                y = np.minimum(np.maximum([math.floor(x_n[i]), math.ceil(x_n[i])], 0), self.inS - 1)
                c, idx = np.unique([x[0] * self.inS + y, x[1] * self.inS + y], return_index=True)
                dist = np.reshape(np.array([np.abs(x - x_n[i]), np.abs(y - y_n[i])]), 4)
                W[i, c] = dist[idx] / sum(dist[idx])
            if series == 1:
                self.W = W
                self.msk = msk
        else:
            msk = []
        # Vectorize the image
        if self.inD:
            image = np.reshape(image, (self.inS ** 2, self.inD))
        else:
            self.inD = 0
            image = np.reshape(image, self.inS ** 2)
        # Sparse matrix multiplication with the original input image to build the new image.
        if series == 1:
            W = self.W
            msk = self.msk
        if self.inD:
            output = np.reshape(W.dot(image), (out_size, out_size, self.inD)).astype(np.uint8)
        else:
            output = np.reshape(W.dot(image), (out_size, out_size))
        return output, msk

    def series_dist(self, in_path, out_path, fov=20, out_size=256, decomp=0, type=1, delete=0, save=1, show=0, masking=1, average =0):
        for i, f_name in enumerate(os.listdir(in_path)):
            print(" Working on image ", i+1)
            file = in_path + '/' + f_name
            img = RetinalCompression.load_image(file)
            if show == 1:
                RetinalCompression.show_image(img)
            img2, msk = RetinalCompression.distort_image(self, img, fov, out_size, decomp, type, series=1)
            if show == 1:
                RetinalCompression.show_image(img2)
            if delete == 1:
                os.remove(file)
            if save == 1:
                print("{}/{}".format(out_path, f_name))
                if masking == 1:
                    if average == 0:
                        plt.imsave("{}/{}".format(out_path, f_name), RetinalCompression.mask(img2, msk, average), cmap='gray', vmin=0, vmax=255)
                else:
                    plt.imsave("{}/{}".format(out_path, f_name), img2, cmap='gray', vmin=0, vmax=255)

