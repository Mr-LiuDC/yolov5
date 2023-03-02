import os
import shutil
import xml.etree.ElementTree as ET
from random import shuffle

from utils.general import download, Path


def convert_label(path, lb_path, ann_name):
    def convert_box(size, box):
        dw, dh = 1. / size[0], 1. / size[1]
        x, y, w, h = (box[0] + box[1]) / 2.0 - 1, (box[2] + box[3]) / 2.0 - 1, box[1] - box[0], box[3] - box[2]
        return x * dw, y * dh, w * dw, h * dh

    in_file = open(path / f'Annotations/{ann_name}')
    out_file = open(lb_path, 'w')
    tree = ET.parse(in_file)
    root = tree.getroot()
    size = root.find('size')
    w = int(size.find('width').text)
    h = int(size.find('height').text)

    # names = list(yaml['names'].values())
    names = ['leconte', 'boerner', 'armandi', 'linnaeus', 'coleoptera', 'acuminatus']
    for obj in root.iter('object'):
        cls = obj.find('name').text
        if cls in names and int(obj.find('difficult').text) != 1:
            xmlbox = obj.find('bndbox')
            bb = convert_box((w, h), [float(xmlbox.find(x).text) for x in ('xmin', 'xmax', 'ymin', 'ymax')])
            cls_id = names.index(cls)  # class id
            out_file.write(" ".join([str(a) for a in (cls_id, *bb)]) + '\n')


val_rate = 0.2
test_rate = 0.1
train_rate = 0.8

data_name = 'Example-VOC'
# base_dir = Path(yaml['path'])
base_dir = Path('../../../datasets')
url = 'https://nutrition-bucket.rplushealth.cn/datasets/'
urls = [f'{url}{data_name}.zip']
download(urls, dir=base_dir, delete=False, curl=True, threads=len(urls))

path = base_dir / f'{data_name}'
lbs_path = base_dir / f'{data_name}' / 'labels'
lbs_path.mkdir(exist_ok=True, parents=True)

ann_files = []
for root, dirs, files in os.walk(base_dir / f'{data_name}' / 'Annotations'):
    ann_files = files

for ann_file in ann_files:
    lb_path = lbs_path / (ann_file.replace('.xml', '.txt'))
    convert_label(path, lb_path, ann_file)

shuffle(ann_files)
shutil.copytree(path / 'JPEGImages', path / 'images')
val_files = ann_files[:round(len(ann_files) * val_rate)]
test_files = ann_files[:round(len(ann_files) * test_rate)]
train_files = ann_files[round(len(ann_files) * val_rate):]
open(f'{path}/yolo_val_list.txt', 'w').write("./images/" + "./images/".join([file.replace('.xml', '.jpg') + '\n' for file in val_files]))
open(f'{path}/yolo_test_list.txt', 'w').write("./images/" + "./images/".join([file.replace('.xml', '.jpg') + '\n' for file in test_files]))
open(f'{path}/yolo_train_list.txt', 'w').write("./images/" + "./images/".join([file.replace('.xml', '.jpg') + '\n' for file in train_files]))
