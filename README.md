![](https://img.shields.io/github/license/ccnmaastricht/ganglion_density_sampling)
![](https://img.shields.io/github/issues/ccnmaastricht/ganglion_density_sampling)
![](https://img.shields.io/github/forks/ccnmaastricht/ganglion_density_sampling)
![](https://img.shields.io/github/stars/ccnmaastricht/ganglion_density_sampling)

**Abstract** The processing of visual information by the human eye can be described as a concatenation of multiple functional stages. Three import stages in this process are the absorption of photonic input by photoreceptors such as rod and cones, the integration of the output of these photoreceptors by bipolar cells, followed by the integration of the output of the bipolar cells by the retinal ganglion Cells. Due to the non-linear distribution of all three cell types along the retina, these processes can be viewed as a sequence of non-linear resampling of the visual input in which each stage builds on the representational output of the previous stage. The distribution of cells, for each cell type, portrays a densely distributed central region, with an incrementally decreasing cell density as the angle of visual eccentricity increases. This results in a high-resolution central vision and a low-resolution representation of peripheral vision. When normalizing the distance between the cells, the representational output will be a wheelbarrow-like distorted image in which the peripheral areas are compressed, compared an intact and high-resolution foveal representation. Overall, the process can be described as non-linear down-sampling method in which the central parts are preserved and the amount of compression for peripheral areas is a function of the visual angle of eccentricity. In deep learning based visual models this process are often overlooked or simplified. Therefore, we propose a biological plausible functional model of non-linear down-sampling. Most computational vision models such as deep convolutional neural networks, start their processing at the level of V1 filtering. As the observation of non-uniform mapping (cortical magnification) of visual input onto the striate cortex is considered to be explained by the retinal ganglion cell density, the distribution of these cells is considered to be the primary functional module that leads to cortical magnification. The currently proposed Retinal Compression Algorithm (RCA) uses the distribution of these cells to compute a biologically plausible approximation of the Ganglion cells’ compression function, and can be considered for initial processing stages of larger computational vision models.

## Requirements:
---
| Package       | Version       |
|:-------------:|:-------------:|
| python       | 3.8.1       |
| numpy         | 1.18.1       |
| scipy    | 1.4.1      |
| matplotlib    | 3.1.3       |


The code is tested on both Windows 10 and Linux operation systems.

## Usage:
Initially, the RCA is initiated as a class/object and can be imported and assigned as any other library class:
```Python
import RCA
RCA1 = RCA.RetinalCompression()
```
### RCA.RetinalCompression.single:

```Python
The RetinalCompression class can be used by calling two simple methods. The first method performs the compression algorithm for a single image:
im = RCA.single(image=None, out_path=None, fov=20, out_size=256, inv=0, type=1, show=1, masking=1, series=0)
```

**Arguments:**
- `image`: 		Array_like. Accepts a Numpy array containing an image, a string link to an image, or is empty. Leaving it empty will call up a GUI to manually select a file.  Default value is set to None.
- `out_path`:	String. Set the path to an output folder for saving the image. If left empty, the resulting image will not be stored. If the output path does not exist yet, it will be created. Default value is set to None.
- `fov`: 		Integer. Field of view coverage, with the distance as visual angle. When decompressing an image, it is advised that the value is set to the fov of the original image. Range=[1, 100]. The Default is set to 20 degrees of visual  angle.
- `out_size`:	Determines the size of the output image in pixels. The value is the size of the output image on one axis. Output image is always a square image. When decompressing an image, it is advised that the value is set to the size of the original image. The default value is set to 256 pixels.
- `inv`:	Integer. Set to 1 to get a decompression of a distorted input image. Set to 0 to have compression. The default value is set to 0.
- `type`:	Integer. Set to 0 for photoreceptor (cones) based distortion, set to 1 for Ganglion cell based distortion. The default value is set to 1 (Ganglion cells).
- `show`:	Integer. Set to 1 to display the image at all intermediate steps. The default value is set to 1.
- `masking`:	Integer. Set to 1 to apply masking the pixels that fall outside of the distortion range. It is recommended to not change this value. The default value is set to 1.
- `series`:		Integer. Set to 1 to store the sparse transformation matrix which can be used for a series of transformations. The default value is set to 0.

**Example:**
```Python
import RCA

# Set custom parameter variables used later as arguments
FOV = 25
source = "C:/<path to image>"

# Script for running the single method. Setting up the class:
RCA1 = RCA.RetinalCompression()

# Calling the method with no image argument. This will call up a GUI to manually select a file.
dIm = RCA1.single(fov=FOV)

# Calling the method with a direct link to an image as the image argument.
dIm = RCA1.single(image=source, fov=FOV)

# Calling the method while forwarding an image directly to the method as the image argument. In this example, decompression of an already distorted image is demonstrated.
rIm = RCA1.single(image=dIm, fov=FOV, inv=1)
```

### RCA.RetinalCompression.series:
The second method to be used is the series method. This method allows the processing of a series of images by forwarding an input folder as a path.

```Python
RCA.series(in_path, out_path, fov=20, out_size=256, inv=0, type=1, show=0, masking=1, series=1)
```

**Arguments:**
- `in_path`:	String. Set the path to an input folder for image loading. All images in the folder will be converted and stored in the out_path folder.

Additional arguments used are similar to those used for the single method. The out_path is now a required argument. In addition, some default values have been changed. Intermediate steps of the process are not shown, and the transformation matrix is saved allowing for faster processing of consecutive images, given that these images are of the same input size.




**Example:**
```Python
import RCA

# Set custom parameter variables used later as arguments.
FOV = 25
in_path = C:/<path to input folder>'
out_path = 'C:/<path to output folder>'

# Script for running the single method. Setting up the class:
RCA = RCA.RetinalCompression()

# Example for calling the method. Only a custom Field of View is passed as an argument.
RCA.series(in_path, out_path, fov=FOV)
```
