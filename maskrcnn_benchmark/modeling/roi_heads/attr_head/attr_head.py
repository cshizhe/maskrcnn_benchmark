# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import torch
from torch import nn
from torch.nn import functional as F

from maskrcnn_benchmark.structures.bounding_box import BoxList
from maskrcnn_benchmark.modeling.matcher import Matcher
from maskrcnn_benchmark.structures.boxlist_ops import boxlist_iou
from maskrcnn_benchmark.modeling.utils import cat

from .roi_attr_predictors import make_roi_attr_predictor
# from .inference import make_roi_mask_post_processor


def keep_only_positive_boxes(boxes):
    """
    Given a set of BoxList containing the `labels` field,
    return a set of BoxList for which `labels > 0`.

    Arguments:
        boxes (list of BoxList)
    """
    assert isinstance(boxes, (list, tuple))
    assert isinstance(boxes[0], BoxList)
    assert boxes[0].has_field("labels")
    positive_boxes = []
    positive_inds = []
    num_boxes = 0
    for boxes_per_image in boxes:
        labels = boxes_per_image.get_field("labels")
        inds_mask = labels > 0
        inds = inds_mask.nonzero().squeeze(1)
        positive_boxes.append(boxes_per_image[inds])
        positive_inds.append(inds_mask)
    return positive_boxes, positive_inds


class ROIAttrHead(torch.nn.Module):
    def __init__(self, cfg):
        super(ROIAttrHead, self).__init__()
        self.cfg = cfg.clone()

        self.predictor = make_roi_attr_predictor(cfg)

        self.proposal_matcher = Matcher(
            cfg.MODEL.ROI_HEADS.FG_IOU_THRESHOLD,
            cfg.MODEL.ROI_HEADS.BG_IOU_THRESHOLD,
            allow_low_quality_matches=False,
        )

        # self.post_processor = make_roi_mask_post_processor(cfg)

    def match_targets_to_proposals(self, proposal, target):
        match_quality_matrix = boxlist_iou(target, proposal)
        matched_idxs = self.proposal_matcher(match_quality_matrix)
        # Attr RCNN needs "labels" and "attributes "fields for creating the targets
        target = target.copy_with_fields(["labels", "attr_labels"])
        # get the targets corresponding GT for each proposal
        # NB: need to clamp the indices because we can have a single
        # GT in the image, and matched_idxs can be -2, which goes
        # out of bounds
        matched_targets = target[matched_idxs.clamp(min=0)]
        matched_targets.add_field("matched_idxs", matched_idxs)
        return matched_targets

    def prepare_targets(self, proposals, targets):
        obj_labels = []
        attr_labels = []
        for proposals_per_image, targets_per_image in zip(proposals, targets):
            matched_targets = self.match_targets_to_proposals(
                proposals_per_image, targets_per_image
            )
            # matched_idxs = matched_targets.get_field("matched_idxs")

            # labels_per_image = matched_targets.get_field("labels")
            # labels_per_image = labels_per_image.to(dtype=torch.int64)

            # # this can probably be removed, but is left here for clarity
            # # and completeness
            # neg_inds = matched_idxs == Matcher.BELOW_LOW_THRESHOLD
            # labels_per_image[neg_inds] = 0

            # # attr scores are only computed on positive samples
            # positive_inds = torch.nonzero(labels_per_image > 0).squeeze(1)

            obj_labels_per_image = matched_targets.get_field('labels')
            attr_labels_per_image = matched_targets.get_field('attr_labels')
            # attr_labels_per_image = attr_labels_per_image[positive_inds]

            obj_labels.append(obj_labels_per_image)
            attr_labels.append(attr_labels_per_image)

        return obj_labels, attr_labels

    def forward(self, features, proposals, targets=None):
        """
        Arguments:
            features (list[Tensor]): feature-maps from possibly several levels
            proposals (list[BoxList]): proposal boxes
            targets (list[BoxList], optional): the ground-truth targets.

        Returns:
            x (Tensor): the result of the feature extractor
            proposals (list[BoxList]): during training, the original proposals
                are returned. During testing, the predicted boxlists are returned
                with the `mask` field set
            losses (dict[Tensor]): During training, returns the losses for the
                head. During testing, returns an empty dict.
        """

        if self.training:
            # during training, only focus on positive boxes
            all_proposals = proposals
            proposals, positive_inds = keep_only_positive_boxes(proposals)

            x = features[torch.cat(positive_inds, dim=0)]

            obj_targets, attr_targets = self.prepare_targets(proposals, targets)
            obj_targets = cat(obj_targets, dim=0)
            attr_targets = cat(attr_targets, dim=0)

            attr_logits = self.predictor(x, obj_targets)
            attr_logprobs = F.log_softmax(attr_logits, dim=1)
            
            loss_attr = - 0.5 * (attr_targets * attr_logprobs)
            loss_attr = torch.mean(torch.sum(loss_attr, dim=1))

            # loss_attr = 0.5 * F.cross_entropy(attr_logits, attr_targets.long())

            return features, all_proposals, dict(loss_attr=loss_attr)
        # if self.training and self.cfg.MODEL.ROI_MASK_HEAD.SHARE_BOX_FEATURE_EXTRACTOR:
        #     x = features
        #     x = x[torch.cat(positive_inds, dim=0)]
        # else:
        #     x = self.feature_extractor(features, proposals)
        

        # if not self.training:
        #     result = self.post_processor(mask_logits, proposals)
        #     return x, result, {}

        


def build_roi_attr_head(cfg):
    return ROIAttrHead(cfg)
