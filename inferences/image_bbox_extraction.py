import os
import argparse

import cv2
import numpy as np
import h5py

import torch
from torchvision import transforms as T
from torchvision.transforms import functional as F

from maskrcnn_benchmark.config import cfg
from maskrcnn_benchmark.modeling.detector import build_detection_model
from maskrcnn_benchmark.utils.checkpoint import DetectronCheckpointer
from maskrcnn_benchmark.structures.image_list import to_image_list

from common import Resize
from common import build_transform
from common import build_detector

'''
if we want few objects, please modify config.yml
_C.MODEL.ROI_HEADS.SCORE_THRESH = 0.05
'''

def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('config_file')
  parser.add_argument('ckpt_file')
  parser.add_argument('image_dir')
  parser.add_argument('name_file')
  parser.add_argument('output_dir')
  parser.add_argument('--layer_name', default='fc7')
  parser.add_argument('--start_id', type=int, default=0)
  parser.add_argument('--end_id', type=int, default=None)
  parser.add_argument('--score_thresh', type=float, default=0.05)
  parser.add_argument('--min_boxes', type=int, default=10)
  opts = parser.parse_args()

  ########### build model #############
  # update the config options with the config file
  cfg.merge_from_file(opts.config_file)
  # manual override some options
  cfg.merge_from_list(['MODEL.DEVICE', 'cuda:0'])#, 'MODEL.ROI_HEADS.SCORE_THRESH', str(opts.score_thresh)])
  cfg.freeze()

  device = torch.device(cfg.MODEL.DEVICE)
  cpu_device = torch.device("cpu")

  model = build_detection_model(cfg)
  model.to(device)
  model.eval()

  checkpointer = DetectronCheckpointer(cfg, model)
  _ = checkpointer.load(f=opts.ckpt_file, use_latest=False)

  transform_fn = build_transform(cfg)

  ########### extract feature #############
  names = np.load(opts.name_file)
  if opts.end_id is None:
    opts.end_id = len(names)
  print('total', opts.end_id - opts.start_id)
  total_images = opts.end_id - opts.start_id

  if not os.path.exists(opts.output_dir):
    os.makedirs(opts.output_dir)

  for i, name in enumerate(names):
    if i < opts.start_id or i >= opts.end_id:
      continue
    outname = name.replace('/', '_')
    outfile = os.path.join(opts.output_dir, '%s.hdf5'%outname)
    if os.path.exists(outfile):
      continue

    img_file = os.path.join(opts.image_dir, name)

    # apply pre-processing to image
    original_image = cv2.imread(img_file)
    height, width = original_image.shape[:-1]
    image = transform_fn(original_image)
    nheight, nwidth = image.size(1), image.size(2)

    # convert to an ImageList, padded so that it is divisible by
    # cfg.DATALOADER.SIZE_DIVISIBILITY
    image_list = to_image_list(image, cfg.DATALOADER.SIZE_DIVISIBILITY)
    image_list = image_list.to(device)

    # compute predictions: one image one mini-batch
    with torch.no_grad():
      # features: tuples in FPN (batch, dim_ft: 256, h, w)
      features = model.forward_backbone(image_list)

      # proposals: list of BoxList (num_proposals:1000)
      # x: tensor, (num_proposals: 1000, dim_ft: 1024)
      proposals = model.forward_rpn(image_list, features)

      detections = model.forward_bbox(features, proposals)

      # result: list of BoxList, (num_bboxes: 100)
      prediction = model.forward_nms(detections, 1601)

      # top 100 bboxes
      result = [o.to(cpu_device) for o in prediction][0]
      result = result.resize((width, height))
      assert len(result) > 0

      scores = result.get_field('scores')
      boxes = result.bbox.data.numpy()
      areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1]) / height / width

      # sort by scores
      _, idx = result.get_field('scores').sort(0, descending=True)
      result = result[idx]

      with h5py.File(outfile, 'w') as outf:
        bbox_fts = result.get_field('features').data.numpy()
        outf.create_dataset(outname, bbox_fts.shape, dtype='float', compression='gzip')
        outf[outname][...] = bbox_fts
        outf[outname].attrs['image_w'] = width
        outf[outname].attrs['image_h'] = height
        outf[outname].attrs['scores'] = result.get_field('scores').data.numpy()
        outf[outname].attrs['labels'] = result.get_field('labels').data.numpy()
        outf[outname].attrs['boxes'] = result.bbox.data.numpy()

      if i % 1000 == 0:
        print('name %s shape %s, processing %d/%d (%.2f%% done)'%(name, 
          bbox_fts.shape, i-opts.start_id, total_images, (i-opts.start_id)*100/total_images))

if __name__ == '__main__':
  main()
