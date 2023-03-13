import os
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
from pycocotools.coco import COCO
from tqdm import tqdm

data_name = 'food-segment'
base_dir = Path('../datasets')
path = base_dir / f'{data_name}'
images_dir = path / 'images'

labels_path = base_dir / f'{data_name}' / 'labels'
labels_path.mkdir(exist_ok=True, parents=True)
masks_path = base_dir / f'{data_name}' / 'annotations'
masks_path.mkdir(exist_ok=True, parents=True)

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
coco = COCO(path / f'annotations.json')

category_ids = coco.getCatIds()
print(f'total categories {len(category_ids)}')
image_ids = coco.getImgIds()
print(f'total images {len(image_ids)}')


def mask2polygons(mask):
    contours, hierarchy = cv2.findContours(mask.astype("uint8"), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    contours_approx = []
    polygons = []
    for contour in contours:
        epsilon = 0.001 * cv2.arcLength(contour, True)
        contour_approx = cv2.approxPolyDP(contour, epsilon, True)
        contours_approx.append(contour_approx)
        polygon = contour_approx.flatten().tolist()
        polygons.append(polygon)
    return polygons


for image_id in tqdm(image_ids, desc='Image processing'):
    image = coco.imgs[image_id]
    input_image = np.array(Image.open(os.path.join(images_dir, image['file_name'])))
    ann_ids = coco.getAnnIds(imgIds=image_id)
    annotations = coco.loadAnns(ann_ids)
    target_mask = np.zeros((image['height'], image['width']))
    for i in range(len(annotations)):
        target_mask += coco.annToMask(annotations[i]) * 255
    plt.imshow(target_mask, cmap='gray')
    plt.show()
    mask_file = str(masks_path / f'{image_id}.png')
    cv2.imwrite(mask_file, target_mask)
    polygons_result = mask2polygons(target_mask)
    line = (np.array(polygons_result).reshape(-1, 2) / np.array([image['width'], image['height']])).reshape(-1).tolist()
    line = [0] + line
    print(line)
