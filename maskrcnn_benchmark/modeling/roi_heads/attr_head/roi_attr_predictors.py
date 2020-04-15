import torch
from torch import nn
from torch.nn import functional as F

from maskrcnn_benchmark.modeling import registry


@registry.ROI_ATTR_PREDICTOR.register("AttrFPNPredictor")
class AttrFPNPredictor(nn.Module):
    def __init__(self, cfg):
        super(AttrFPNPredictor, self).__init__()
        num_obj_classes = cfg.MODEL.ROI_BOX_HEAD.NUM_CLASSES
        num_attr_classes = cfg.MODEL.ROI_ATTR_HEAD.NUM_ATTR_CLASSES

        self.obj_embedding = nn.Embedding(num_obj_classes, cfg.MODEL.ROI_ATTR_HEAD.OBJ_EMBED_SIZE)
        self.fc_attr = nn.Linear(cfg.MODEL.ROI_BOX_HEAD.MLP_HEAD_DIM + cfg.MODEL.ROI_ATTR_HEAD.OBJ_EMBED_SIZE, 
                                 cfg.MODEL.ROI_ATTR_HEAD.ATTR_HIDDEN_SIZE)
        self.logit_attr = nn.Linear(cfg.MODEL.ROI_ATTR_HEAD.ATTR_HIDDEN_SIZE, num_attr_classes)

        nn.init.normal_(self.obj_embedding.weight, mean=0, std=0.01)
        nn.init.normal_(self.fc_attr.weight, mean=0, std=0.01)
        nn.init.normal_(self.logit_attr.weight, mean=0, std=0.01)

        for l in [self.fc_attr, self.logit_attr]:
            nn.init.constant_(l.bias, 0)

    def forward(self, x, obj_classes):
        '''Args:
        - x: (batch, hidden_size)
        - obj_classes: (batch, )
        '''
        if x.ndimension() == 4:
            assert list(x.shape[2:]) == [1, 1]
            x = x.view(x.size(0), -1)
        obj_embeds = self.obj_embedding(obj_classes.long())
        attr_hiddens = self.fc_attr(torch.cat([x, obj_embeds], 1))
        attr_logits = self.logit_attr(F.relu(attr_hiddens, inplace=True))
        return attr_logits


def make_roi_attr_predictor(cfg):
    func = registry.ROI_ATTR_PREDICTOR[cfg.MODEL.ROI_ATTR_HEAD.PREDICTOR]
    return func(cfg)
