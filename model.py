import numpy as np
import torch
import torch.nn as nn
import torchvision
import torch.nn.functional as F

class CAMModel(nn.Module):
    def __init__(self, args):
        super(CAMModel, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.args = args

        ###################################################
        rennet18 = torchvision.models.resnet18(pretrained=True)
        layers_rennet18 = list(rennet18.children())[:6]
        self.backbone_layer = nn.Sequential(*layers_rennet18)

        self.GAP = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.W_cls = nn.Linear(in_features=128, out_features=10)
        ###################################################

    def forward(self, img_L, img_S=None):

        batch_size = img_L.shape[0] # 64
        if self.args.mode == 'CAM':
            # batch_size x 3 x 224 x 224 --> batch_size x 128 x 28 x 28
            f_L = self.backbone_layer(img_L)
            # batch_size x 128 x 128 x 128 --> batch_size x 128 x 1 x 1
            out_GAP_L = self.GAP(f_L)
            # batch_size x 128 x 1 x 1 --> batch_size x 128
            w_L = out_GAP_L.squeeze()
            # batch_size x 128 --> batch_size x 10
            w_cls_L = self.W_cls(w_L)

            return w_cls_L, (f_L, w_L)

        elif self.args.mode == 'SEG':
            # batch_size x 128 x 28 x 28
            f_L = self.backbone_layer(img_L)
            # batch_size x 128 x 1 x 1
            out_GAP_L = self.GAP(f_L)
            # batch_size x 128
            w_L = out_GAP_L.squeeze()
            # batch_size x 10
            w_cls_L = self.W_cls(w_L)

            if img_S is not None:
                # batch_size x 128 x 14 x 14
                f_S = self.backbone_layer(img_S)
                # batch_size x 128 x 1 x 1
                out_GAP_S = self.GAP(f_S)
                # batch_size x 128
                w_S = out_GAP_S.squeeze()
                # batch_size x 10
                w_cls_S = self.W_cls(w_S)

                return w_cls_S, w_cls_L, f_L, f_S, w_L, w_S
            else:
                return w_cls_L, (f_L, w_L)
        else:
            # batch_size x 128 x 28 x 28
            f_L = self.backbone_layer(img_L)
            # batch_size x 128 x 1 x 1
            out_GAP_L = self.GAP(f_L)
            # batch_size x 128
            w_L = out_GAP_L.squeeze()
            # batch_size x 10
            w_cls_L = self.W_cls(w_L)

            if img_S is not None:
                # batch_size x 128 x 28 x 28
                f_S = self.backbone_layer(img_S)
                # batch_size x 128 x 1 x 1
                out_GAP_S = self.GAP(f_S)
                # batch_size x 128
                w_S = out_GAP_S.squeeze()
                # batch_size x 10
                w_cls_S = self.W_cls(w_S)

                return w_cls_S, w_cls_L, f_L, f_S, w_L, w_S
            else:
                return w_cls_L, (f_L, w_L)


