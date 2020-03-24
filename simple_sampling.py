import numpy as np 
from math import floor, ceil
from scipy.sparse import lil_matrix as sparse
from scipy.special import lambertw

class RetinalCompression:
    def __init__(self):
        self.a = .98  # The weighting of the first term in the original equation
        self.r2 = 1.05  # The eccentricity at which density is reduced by a factor of four (and spacing is doubled)
        self.re = 22  # Scale factor of the exponential. Not used in our version.
        self.dg = 33162  # Cell density at r = 0
        self.C = (3.4820e+04 + .1)  # Constant added to the integral to make sure that if x == 0, y == 0.
        self.size_in, self.inR, self.inC, self.inD = int, int, int, int  # Dimensions of the selected image (pixel space)
        self.bg_color = 127  # Background color for the generated image.
        self.num_pixels_in = [] # Placeholder for number of pixels of input image
        self.size_in = []
        self.size_out = []
        self.W = None  # Placeholder for spare matrix used for image transformation when using series_dist method
        self.mask = None  # Placeholder for mask when using series_dist method

    def fi(self, r):
        # Integrated Ganglion Density formula (3), taken from Watson (2014). Maps from degrees of visual angle to the
        # amount of cells.
        return self.C - np.divide((np.multiply(self.dg, self.r2 ** 2)), (r + self.r2))

    def fii(self, r):
        # Inverted integrated Ganglion Density formula (3), taken from Watson (2014).
        return np.divide(np.multiply(self.dg, self.r2 ** 2), (self.C - r)) - self.r2

    def create_mapping(self, parameters):
        # Arguments:
        #     parameters = dictionary
       
        eccentricity = parameters['field_of_view'] / 2
        radius_in = parameters['resolution_in'] / 2

        self.size_in = parameters['resolution_in']
        self.size_out = parameters['resolution_out']
        self.num_pixels_in = self.size_in**2
        num_pixels_out = self.size_out**2

        num_cells = self.fi(eccentricity)
        cells_per_pixel = num_cells / radius_in
        visual_field = np.linspace(-eccentricity, eccentricity, self.size_out)
        vf_x, vf_y = np.meshgrid(visual_field, visual_field)
        vf_x = np.reshape(vf_x, num_pixels_out)
        vf_y = np.reshape(vf_y, num_pixels_out)
        
        vf_angle = np.angle(vf_x + vf_y * 1j)
        vf_radius = np.abs(vf_x + vf_y * 1j)

        self.mask = vf_radius <= num_cells

        new_radius = self.fii(vf_radius) / cells_per_pixel

        new_x = np.multiply(np.cos(vf_angle), new_radius) + radius_in
        new_y = np.multiply(np.sin(vf_angle), new_radius) + radius_in


        self.W = sparse((num_pixels_out, self.num_pixels_in), dtype=np.float)
        # Sometimes division by 0 might happen. This line makes sure the user won't see a warning when this happens.
        np.seterr(divide='ignore', invalid='ignore')
        for i in range(num_pixels_out):
            # Pixel indices will almost always not be a perfect integer value. Therefore, the value of the new pixel
            # is value is determined by taking the average of all pixels involved. E.g. a value of 4.3 is converted
            # to the indices 4, and 5. The RGB values are weighted accordingly (0.7 for index 4, and 0.3 for index
            # 5). Additionally, boundary checking is used. Values can never be smaller than 0, or larger than the
            # maximum index of the image.
            x = np.minimum(np.maximum([floor(new_y[i]), ceil(new_y[i])], 0), self.size_in - 1)
            y = np.minimum(np.maximum([floor(new_x[i]), ceil(new_x[i])], 0), self.size_in - 1)
            c, idx = np.unique([x[0] * self.size_in + y, x[1] * self.size_in + y], return_index=True)
            dist = np.reshape(np.array([np.abs(x - new_x[i]), np.abs(y - new_y[i])]), 4)
            self.W[i, c] = dist[idx] / sum(dist[idx])
    
    def distort_image(self, image):
        try:
            depth =  np.size(img, axis=2) # Determine dimensions of the selected image (pixel space)
        except:
            depth = None

        if depth:
            image = np.reshape(image, (self.num_pixels_in, depth))
            output = np.reshape(np.dot(self.W, image), (self.size_out, self.size_out, depth)).astype(np.uint8)
        else:
            output = np.reshape(np.dot(self.W, image), (self.size_out, self.size_out)).astype(np.uint8)

        return output


            