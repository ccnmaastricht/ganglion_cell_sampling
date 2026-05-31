"""Compress and decompress a single image.

If examples/images/ is absent or empty, the example image is downloaded
automatically from Wikimedia Commons.
"""

import os
import sys
import urllib.request

import cv2
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from retinal_sampling import RetinalSampler

# ---------------------------------------------------------------------------
# Image retrieval
# ---------------------------------------------------------------------------

IMAGES_DIR = os.path.join(os.path.dirname(__file__), 'images')

_URL = (
    'https://upload.wikimedia.org/wikipedia/commons/a/ad/'
    'Parc_de_Versailles%2C_bosquet_des_Bains-d%27Apollon%2C_grotte_01.jpg'
)
_FILENAME = 'versailles.jpg'


def _ensure_image() -> str:
    os.makedirs(IMAGES_DIR, exist_ok=True)
    fpath = os.path.join(IMAGES_DIR, _FILENAME)
    if not os.path.isfile(fpath):
        print(f'Downloading {_FILENAME} ...')
        req = urllib.request.Request(_URL, headers={'User-Agent': 'retinal-sampling-example/1.0'})
        with urllib.request.urlopen(req) as resp, open(fpath, 'wb') as f:
            f.write(resp.read())
    return fpath


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

MIN_SIZE = 1024   # upscale if the cropped square is smaller than this
MAX_SIZE = 2048   # downscale if the cropped square is larger than this

image_path = _ensure_image()
image = cv2.imread(image_path)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Center-crop to square, then clamp to [MIN_SIZE, MAX_SIZE].
h, w = image.shape[:2]
size = min(h, w)
row_off = (h - size) // 2
col_off = (w - size) // 2
image = image[row_off : row_off + size, col_off : col_off + size]

target = min(max(size, MIN_SIZE), MAX_SIZE)
if size != target:
    interp = cv2.INTER_CUBIC if target > size else cv2.INTER_AREA
    image = cv2.resize(image, (target, target), interpolation=interp)

input_size = image.shape[0]

sampler = RetinalSampler(cell_type='ganglion')
sampler.build_mapping(input_size=input_size, output_size=1024, fov=10.0)

intermediate  = sampler.compress(image)
final         = sampler.crop(intermediate)
decompressed  = sampler.decompress(intermediate)

print('input shape       :', image.shape)
print('intermediate shape:', intermediate.shape)
print('final shape       :', final.shape)
print('decompressed shape:', decompressed.shape)

plt.figure(figsize=(12, 4))
plt.subplot(1, 3, 1)
plt.title('Original')
plt.imshow(image)
plt.axis('off')
plt.subplot(1, 3, 2)
plt.title('Retinal sampling')
plt.imshow(final)
plt.axis('off')
plt.subplot(1, 3, 3)
plt.title('Decompressed')
plt.imshow(decompressed)
plt.axis('off')
plt.tight_layout()
plt.show()
