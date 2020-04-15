import os
import json
import numpy as np
from PIL import Image

import torch
import torch.utils.data

from maskrcnn_benchmark.structures.bounding_box import BoxList

class VGDataset(torch.utils.data.Dataset):
  def __init__(self, data_dir, split, image_dir, transforms=None):
    '''
    Args:
      - data_dir: 
        - image_image names for dataset splitation
        - vocab_dir: objects_vocab.txt, attributes_vocab.txt
          format: one class per line (similar words are splitted by comma)
      - split: trn, val, tst
      - sg_file: format [
        {
          'image_id': int,
          'objects': [
            {
              'name': str, 'object_id': int, 
              'x': int, 'y': int, 'w': int, 'h': int,
              'attributes': [str, str, ...]
            }
          ]
          'relationships': [
            {
              'predicate': str, 'object_id': int, 'subject_id': int
            }
          ]
        }
      ]
      - image_dir: contain 'image_id'.jpg
    '''
    super().__init__()

    self.img_names = np.load(os.path.join(data_dir, '%s_names.npy'%split))

    self.obj_categories = ['__background__'] + [line.strip() \
      for line in open(os.path.join(data_dir, '1600-400-20', 'objects_vocab.txt')).readlines()]
    self.attr_categories = ['__no_attribute__'] + [line.strip() \
      for line in open(os.path.join(data_dir, '1600-400-20', 'attributes_vocab.txt')).readlines()]

    self.obj_name2int, self.attr_name2int = {}, {}
    for i, line in enumerate(self.obj_categories):
      for token in line.split(','):
        self.obj_name2int[token] = i
    for i, line in enumerate(self.attr_categories):
      for token in line.split(','):
        self.attr_name2int[token] = i

    #aliasfile = os.path.join(data_dir, 'object_alias.txt')
    #with open(aliasfile) as f:
    #  for line in f:
    #    tokens = line.strip().split(',')
    #    if tokens[0] in self.obj_name2int:
    #      for token in tokens[1:]:
    #        self.obj_name2int[token] = self.obj_name2int[tokens[0]]

    with open(os.path.join(data_dir, 'scene_graphs.json')) as f:
      self.sg_data = json.load(f)
      id2imgname = {int(imgname.split('.')[0]): imgname for imgname in self.img_names}
      self.sg_data = [x for x in self.sg_data if x['image_id'] in id2imgname]
      
      filtered_sg_data = []
      for x in self.sg_data:
        img_objs = [o for o in x['objects'] if o['name'] in self.obj_name2int]
        if len(img_objs) > 0:
          x['objects'] = img_objs
          for k, obj in enumerate(x['objects']):
            x['objects'][k]['attributes'] = [attr for attr in obj['attributes'] if attr in self.attr_name2int]
          filtered_sg_data.append(x)
      self.sg_data = filtered_sg_data
      self.img_names = [id2imgname[x['image_id']] for x in self.sg_data]

    print('num images', len(self.img_names))
    self.image_dir = image_dir
    self.transforms = transforms

  def __len__(self):
    return len(self.img_names)

  def __getitem__(self, idx):
    # load the image as a PIL Image
    img_name = self.img_names[idx]
    img = Image.open(os.path.join(self.image_dir, img_name)).convert('RGB')

    target = self.get_groundtruth(idx)
    target = target.clip_to_image(remove_empty=True)

    if self.transforms is not None:
      img, target = self.transforms(img, target)

    return img, target, idx

  def get_groundtruth(self, idx):
    anno = self._preprocess_annotation(idx)

    target = BoxList(anno['boxes'], (anno['width'], anno['height']), mode='xyxy')
    target.add_field('labels', anno['labels'])
    target.add_field('attr_labels', anno['attr_labels'])
    return target

  def _preprocess_annotation(self, idx):
    anno = self.sg_data[idx]
    width, height = anno['width'], anno['height']

    boxes, labels, attr_labels = [], [], []
    for obj in anno['objects']:
      box = [
          max(0, obj['x']), 
          max(0, obj['y']), 
          min(width - 1, obj['x'] + obj['w']), 
          min(height - 1, obj['y'] + obj['h'])
        ]
      if box[2] < box[0] or box[3] < box[1]:
        box = [0, 0, width - 1, height - 1]
      boxes.append(box)

      labels.append(self.obj_name2int[obj['name']])
      attr_label = [0 for _ in range(len(self.attr_categories))]
      for attr_name in obj.get('attributes', [])[:16]:
        attr_label[self.attr_name2int[attr_name]] = 1
      attr_labels.append(attr_label)

    return {
      'boxes': torch.tensor(boxes, dtype=torch.float32),
      'labels': torch.tensor(labels),
      'attr_labels': torch.tensor(attr_labels),
      'height': anno['height'],
      'width': anno['width'],
    }

  def get_img_info(self, idx):
    # get img_height and img_width. This is used if
    # we want to split the batches according to the aspect ratio
    # of the image, as it can be more efficient than loading the
    # image from disk
    x = self.sg_data[idx]
    return {'height': x['height'], 'width': x['width']}

