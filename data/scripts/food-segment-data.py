import json

import cv2

from utils.general import download, Path

val_rate = 0.2
test_rate = 0.1
train_rate = 0.8

data_name = 'D0001-segment-example'
base_dir = Path('../../../datasets')
path = base_dir / f'{data_name}'
lbs_path = base_dir / f'{data_name}' / 'labels'
lbs_path.mkdir(exist_ok=True, parents=True)
url = 'https://nutrition-bucket.rplushealth.cn/datasets/'
urls = [f'{url}{data_name}.zip']
download(urls, dir=path, delete=False, curl=True, threads=len(urls))

# with open(path / f'annotations.json') as data:
#     json_data = json.load(data)
# images_info = json_data['images']
# categories_info = json_data['categories']
# annotations_info = json_data['annotations']
# for image_info in images_info:
#     img_id = image_info['id']
#     segmentations = [segmentation for segmentation in annotations_info if segmentation['image_id'] == img_id]
#     print(len(segmentations))

from pycocotools.coco import COCO
import os
from PIL import Image
import numpy as np
from matplotlib import pyplot as plt

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
coco = COCO(path / f'annotations.json')
img_dir = path / 'JPEGImages'
image_id = 13

img = coco.imgs[image_id]
image = np.array(Image.open(os.path.join(img_dir, img['file_name'])))
plt.imshow(image)
plt.show()


plt.imshow(image)
cat_ids = coco.getCatIds()
anns_ids = coco.getAnnIds(imgIds=img['id'], catIds=cat_ids, iscrowd=None)
anns = coco.loadAnns(anns_ids)
coco.showAnns(anns)
plt.show()


mask = np.zeros((img['height'], img['width']))
for i in range(len(anns)):
    mask += coco.annToMask(anns[i])*255
plt.imshow(mask)
plt.show()
plt.imshow(mask, cmap='gray')
plt.show()
cv2.imwrite(path / 'labels' / f'{image_id}.png', mask)
