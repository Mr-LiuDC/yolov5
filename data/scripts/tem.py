import json
import os
from pathlib import Path
from random import shuffle

import numpy as np
from tqdm import tqdm

from utils.general import download


def convert_coco_json_to_yolo_txt(output_path, json_file, use_segments=True):
    df_img_id = []
    df_img_name = []
    df_img_width = []
    df_img_height = []
    with open(json_file) as jf:
        json_data = json.load(jf)

    label_file = os.path.join(output_path, "labels.txt")
    with open(label_file, "w") as f:
        for image in tqdm(json_data["images"], desc="Annotation txt for each iamge"):
            img_id = image["id"]
            img_name = os.path.basename(image["file_name"])
            img_width = image["width"]
            img_height = image["height"]
            df_img_id.append(img_id)
            df_img_name.append(img_name)
            df_img_width.append(img_width)
            df_img_height.append(img_height)

            anno_in_image = [anno for anno in json_data["annotations"] if anno["image_id"] == img_id]
            anno_txt = os.path.join(f'{output_path}/labels', img_name.split(".")[0] + ".txt")

            h, w, f = image['height'], image['width'], image['file_name']
            bboxes = []
            segments = []
            with open(anno_txt, "w") as f:
                for anno in anno_in_image:
                    # x, y, w, h = convert_bbox_coco2yolo(img_width, img_height, bbox_COCO)
                    if anno['iscrowd']:
                        continue
                    # The COCO box format is [top left x, top left y, width, height]
                    box = np.array(anno['bbox'], dtype=np.float64)
                    box[:2] += box[2:] / 2  # xy top-left corner to center
                    box[[0, 2]] /= w  # normalize x
                    box[[1, 3]] /= h  # normalize y
                    if box[2] <= 0 or box[3] <= 0:  # if w <= 0 and h <= 0
                        continue
                    # cls = anno['category_id'] - 1
                    cls = anno['n_id'] - 1
                    box = [cls] + box.tolist()
                    if box not in bboxes:
                        bboxes.append(box)
                    # Segments
                    if use_segments:
                        if len(anno['segmentation']) > 1:
                            s = merge_multi_segment(anno['segmentation'])
                            s = (np.concatenate(s, axis=0) / np.array([w, h])).reshape(-1).tolist()
                        else:
                            s = [j for i in anno['segmentation'] for j in i]  # all segments concatenated
                            s = (np.array(s).reshape(-1, 2) / np.array([w, h])).reshape(-1).tolist()
                        s = [cls] + s
                        if s not in segments:
                            segments.append(s)

                    last_iter = len(bboxes) - 1
                    line = *(segments[last_iter] if use_segments else bboxes[last_iter]),  # cls, box or segments
                    f.write(('%g ' * len(line)).rstrip() % line + '\n')

    categories = []
    for category in tqdm(json_data["categories"], desc="Categories"):
        category_name = category["name"]
        categories.append(category_name)
    categories = sorted(categories)
    categories = sorted(categories, key=len)
    with open(label_file, "w") as file:
        for line in categories:
            file.write("%s\n" % line)


def merge_multi_segment(segments):
    s = []
    segments = [np.array(i).reshape(-1, 2) for i in segments]
    idx_list = [[] for _ in range(len(segments))]

    # record the indexes with min distance between each segment
    for i in range(1, len(segments)):
        idx1, idx2 = min_index(segments[i - 1], segments[i])
        idx_list[i - 1].append(idx1)
        idx_list[i].append(idx2)

    # use two round to connect all the segments
    for k in range(2):
        # forward connection
        if k == 0:
            for i, idx in enumerate(idx_list):
                # middle segments have two indexes
                # reverse the index of middle segments
                if len(idx) == 2 and idx[0] > idx[1]:
                    idx = idx[::-1]
                    segments[i] = segments[i][::-1, :]

                segments[i] = np.roll(segments[i], -idx[0], axis=0)
                segments[i] = np.concatenate([segments[i], segments[i][:1]])
                # deal with the first segment and the last one
                if i in [0, len(idx_list) - 1]:
                    s.append(segments[i])
                else:
                    idx = [0, idx[1] - idx[0]]
                    s.append(segments[i][idx[0]:idx[1] + 1])

        else:
            for i in range(len(idx_list) - 1, -1, -1):
                if i not in [0, len(idx_list) - 1]:
                    idx = idx_list[i]
                    nidx = abs(idx[1] - idx[0])
                    s.append(segments[i][nidx:])
    return s


def min_index(arr1, arr2):
    dis = ((arr1[:, None, :] - arr2[None, :, :]) ** 2).sum(-1)
    return np.unravel_index(np.argmin(dis, axis=None), dis.shape)


val_rate = 0.2
test_rate = 0.1
train_rate = 0.8

data_name = 'example'
# base_dir = Path(yaml['path'])
base_dir = Path('../../../datasets')
url = 'https://nutrition-bucket.rplushealth.cn/datasets/'
urls = [f'{url}{data_name}.zip']
download(urls, dir=base_dir, delete=False, curl=True, threads=len(urls))

path = base_dir / f'{data_name}'
lbs_path = base_dir / f'{data_name}' / 'labels'
lbs_path.mkdir(exist_ok=True, parents=True)

convert_coco_json_to_yolo_txt(path, f'{path}/annotations-new.json', True)

ann_files = []
for root, dirs, files in os.walk(base_dir / f'{data_name}' / 'labels'):
    ann_files = files

shuffle(ann_files)
val_files = ann_files[:round(len(ann_files) * val_rate)]
test_files = ann_files[:round(len(ann_files) * test_rate)]
train_files = ann_files[round(len(ann_files) * val_rate):]
open(f'{path}/yolo_val_list.txt', 'w').write("./images/" + "./images/".join([file.replace('.txt', '.jpg') + '\n' for file in val_files]))
open(f'{path}/yolo_test_list.txt', 'w').write("./images/" + "./images/".join([file.replace('.txt', '.jpg') + '\n' for file in test_files]))
open(f'{path}/yolo_train_list.txt', 'w').write("./images/" + "./images/".join([file.replace('.txt', '.jpg') + '\n' for file in train_files]))
