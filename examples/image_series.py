"""Apply retinal sampling to a folder of images.

build_mapping is called once; the same precomputed indices and weights are
reused for every image, so the per-image cost is four numpy index lookups and
a weighted sum.

If examples/images/ is absent or empty, three example images are downloaded
automatically from Wikimedia Commons.
"""

import os
import sys
import urllib.request

import cv2

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from retinal_sampling import RetinalSampler

# ---------------------------------------------------------------------------
# Image retrieval
# ---------------------------------------------------------------------------

IMAGES_DIR = os.path.join(os.path.dirname(__file__), 'images')

_URLS = {
    'versailles.jpg': (
        'https://upload.wikimedia.org/wikipedia/commons/a/ad/'
        'Parc_de_Versailles%2C_bosquet_des_Bains-d%27Apollon%2C_grotte_01.jpg'
    ),
    'absis_burgal.jpg': (
        'https://upload.wikimedia.org/wikipedia/commons/6/6b/'
        'MNAC_-_Absis_del_Burgal_-_pano_01.jpg'
    ),
    'palazzo_vincentini.jpg': (
        'https://upload.wikimedia.org/wikipedia/commons/f/ff/'
        'Palazzo_Vincentini_-_giardini%2C_panorama_1.jpg'
    ),
}


def _ensure_images() -> None:
    os.makedirs(IMAGES_DIR, exist_ok=True)
    if any(True for f in os.listdir(IMAGES_DIR) if f.lower().endswith('.jpg')):
        return
    for fname, url in _URLS.items():
        print(f'Downloading {fname} ...')
        req = urllib.request.Request(url, headers={'User-Agent': 'retinal-sampling-example/1.0'})
        dest = os.path.join(IMAGES_DIR, fname)
        with urllib.request.urlopen(req) as resp, open(dest, 'wb') as f:
            f.write(resp.read())


# ---------------------------------------------------------------------------
# Sampling parameters
# ---------------------------------------------------------------------------

MIN_SIZE    = 1024  # upscale if the cropped square is smaller than this
MAX_SIZE    = 2048  # downscale if the cropped square is larger than this
OUTPUT_SIZE = 256
FOV         = 20.0
OUT_DIR     = os.path.join(IMAGES_DIR, 'sampled')

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

_ensure_images()
os.makedirs(OUT_DIR, exist_ok=True)

image_files = sorted(
    f for f in os.listdir(IMAGES_DIR) if f.lower().endswith('.jpg')
)

sampler = None

for fname in image_files:
    image = cv2.imread(os.path.join(IMAGES_DIR, fname))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

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

    # Build mapping on first image; rebuild if input size changes.
    if sampler is None or sampler._input_size != input_size:
        sampler = RetinalSampler(cell_type='ganglion')
        sampler.build_mapping(input_size=input_size, output_size=OUTPUT_SIZE, fov=FOV)

    sampled = sampler.crop(sampler.compress(image))

    out_path = os.path.join(OUT_DIR, fname)
    cv2.imwrite(out_path, cv2.cvtColor(sampled, cv2.COLOR_RGB2BGR))
    print(f'{fname}: {image.shape[:2]} → {sampled.shape[:2]}')
