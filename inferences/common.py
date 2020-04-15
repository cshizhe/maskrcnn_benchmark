import copy
import numpy as np

import torch
from torchvision import transforms as T
from torchvision.transforms import functional as F

from maskrcnn_benchmark.config import cfg
from maskrcnn_benchmark.modeling.detector import build_detection_model
from maskrcnn_benchmark.utils.checkpoint import DetectronCheckpointer



class Resize(object):
  def __init__(self, min_size, max_size):
    self.min_size = min_size
    self.max_size = max_size

  # modified from torchvision to add support for max size
  def get_size(self, image_size):
    w, h = image_size
    size = self.min_size
    max_size = self.max_size
    if max_size is not None:
      min_original_size = float(min((w, h)))
      max_original_size = float(max((w, h)))
      if max_original_size / min_original_size * size > max_size:
        size = int(round(max_size * min_original_size / max_original_size))

    if (w <= h and w == size) or (h <= w and h == size):
      return (h, w)

    if w < h:
      ow = size
      oh = int(size * h / w)
    else:
      oh = size
      ow = int(size * w / h)

    return (oh, ow)

  def __call__(self, image):
    size = self.get_size(image.size)
    image = F.resize(image, size)
    return image


def build_transform(cfg):
  """
  Creates a basic transformation that was used to train the models
  """

  # we are loading images with OpenCV, so we don't need to convert them
  # to BGR, they are already! So all we need to do is to normalize
  # by 255 if we want to convert to BGR255 format, or flip the channels
  # if we want it to be in RGB in [0-1] range.
  if cfg.INPUT.TO_BGR255:
    to_bgr_transform = T.Lambda(lambda x: x * 255)
  else:
    to_bgr_transform = T.Lambda(lambda x: x[[2, 1, 0]])

  normalize_transform = T.Normalize(
    mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD
  )
  min_size = cfg.INPUT.MIN_SIZE_TEST
  max_size = cfg.INPUT.MAX_SIZE_TEST
  transform = T.Compose(
    [
      T.ToPILImage(),
      Resize(min_size, max_size),
      T.ToTensor(),
      to_bgr_transform,
      normalize_transform,
    ]
  )
  return transform


def build_detector(config_file, ckpt_file):
  # update the config options with the config file
  cfg.merge_from_file(config_file)
  # manual override some options
  cfg.merge_from_list(['MODEL.DEVICE', 'cuda:0', 'MODEL.ROI_HEADS.SCORE_THRESH', 0.0])
  cfg.freeze()

  device = torch.device(cfg.MODEL.DEVICE)

  model = build_detection_model(cfg)
  model.to(device)
  model.eval()

  checkpointer = DetectronCheckpointer(cfg, model)
  _ = checkpointer.load(f=ckpt_file, use_latest=False)

  return model, cfg


