"""
Basic OSTrack model.
"""
import math
import os
from typing import List

import torch
from torch import nn
from torch.nn.modules.transformer import _get_clones

from lib.models.layers.head import build_box_head
from lib.models.ostrack.vit import vit_base_patch16_224
from lib.models.ostrack.vit_ce import vit_large_patch16_224_ce, vit_base_patch16_224_ce
from lib.utils.box_ops import box_xyxy_to_cxcywh
from lib.models.language_model import build_bert
from .positional_encoding.untied.absolute import Untied2DPositionalEncoder
from .cross_attention import CrossAttentionBlock


class OSTrack(nn.Module):
    """ This is the base class for OSTrack """

    def __init__(self, transformer, language_backbone,box_head, aux_loss=False, head_type="CORNER"):
        """ Initializes the model.
        Parameters:
            transformer: torch module of the transformer architecture.
            aux_loss: True if auxiliary decoding losses (loss at each decoder layer) are to be used.
        """
        super().__init__()
        self.backbone = transformer
        self.language_backbone = language_backbone
        self.box_head = box_head

        self.aux_loss = aux_loss
        self.head_type = head_type
        if head_type == "CORNER" or head_type == "CENTER":
            self.feat_sz_s = int(box_head.feat_sz)
            self.feat_len_s = int(box_head.feat_sz ** 2)

        if self.aux_loss:
            self.box_head = _get_clones(self.box_head, 6)

        self.proj_1 = nn.Conv2d(in_channels=768*3,out_channels=768,kernel_size=3,stride=1,padding=1,bias=True)
        self.atten_1 = nn.MultiheadAttention(768,8,dropout=0.1,bias=True,batch_first=True)
        self.relu = nn.ReLU()
        self.norm_1 = nn.LayerNorm(768,eps=1e-6)
        self.norm_2 = nn.LayerNorm(768, eps=1e-6)
        self.untied_text_pos_enc_p = Untied2DPositionalEncoder(dim=768, num_heads=8, h=6, w=6, with_q=False)
        self.untied_search_pos_enc_p = Untied2DPositionalEncoder(dim=768, num_heads=8, h=16, w=16, with_k=False)
        self.cross_attention = CrossAttentionBlock(dim=768, num_heads=8)
        self.proj_2 = nn.Conv2d(in_channels=768 * 2, out_channels=768, kernel_size=3, stride=1, padding=1, bias=True)
        self.norm_3 = nn.LayerNorm(768, eps=1e-6)


    def forward(self, template: torch.Tensor,
                search: torch.Tensor,
                text,
                ce_template_mask=None,
                ce_keep_rate=None,
                return_last_attn=False,
                ):
        text_fea = self.language_backbone(text)
        x, aux_dict = self.backbone(z=template, x=search,text=text_fea,
                                    ce_template_mask=ce_template_mask,
                                    ce_keep_rate=ce_keep_rate,
                                    return_last_attn=return_last_attn, )

        # Forward head
        feat_last = x
        # if isinstance(x, list):
        #     feat_last = x[-1]
        out = self.forward_head(text_fea, feat_last, None)

        out.update(aux_dict)
        out['backbone_feat'] = x
        return out

    def forward_head(self, text_fea, cat_feature, gt_score_map=None):
        """
        cat_feature: output embeddings of the backbone, it can be (HW1+HW2, B, C) or (HW2, B, C)
        """
        b = text_fea.shape[0]
        x1 = cat_feature[0][:, -self.feat_len_s:]
        x2 = cat_feature[1][:, -self.feat_len_s:]
        x3 = cat_feature[2][:, -self.feat_len_s:]
        x_one = torch.concat((x1,x2,x3),dim=1).permute(0,2,1).contiguous().view(b,-1,16,16)
        x_one = self.proj_1(x_one).view(b,768,256).permute(0,2,1).contiguous()
        # x_one,_ = self.atten_1(x_one,x_one,x_one)
        x_one1, _ = self.atten_1(x1, x2, x3)
        x_one = self.norm_1(self.relu(x_one + x_one1))
        # x_one = x1 + x2 + x3
        untied_text_pos_enc_k = self.untied_text_pos_enc_p()
        untied_search_pos_enc_p_q = self.untied_search_pos_enc_p()
        attn_pos_enc_p = (untied_search_pos_enc_p_q @ (untied_text_pos_enc_k.transpose(-2, -1))).unsqueeze(0)
        x_two = self.cross_attention(q=x3, kv=text_fea,  q_ape=None, k_ape=None,attn_pos=attn_pos_enc_p)
        x_two = self.norm_2(self.relu(x_two))

        x_three = torch.concat((x_one,x_two),dim=2).permute(0,2,1).contiguous().view(b,-1,16,16)
        x_three = self.proj_2(x_three).view(b,768,256).permute(0,2,1).contiguous()
        enc_opt = self.norm_3(self.relu(x_three)+ x3)

        # enc_opt = cat_feature[:, -self.feat_len_s:]  # encoder output for the search region (B, HW, C)
        opt = (enc_opt.unsqueeze(-1)).permute((0, 3, 2, 1)).contiguous()
        bs, Nq, C, HW = opt.size()
        opt_feat = opt.view(-1, C, self.feat_sz_s, self.feat_sz_s)

        if self.head_type == "CORNER":
            # run the corner head
            pred_box, score_map = self.box_head(opt_feat, True)
            outputs_coord = box_xyxy_to_cxcywh(pred_box)
            outputs_coord_new = outputs_coord.view(bs, Nq, 4)
            out = {'pred_boxes': outputs_coord_new,
                   'score_map': score_map,
                   }
            return out

        elif self.head_type == "CENTER":
            # run the center head
            score_map_ctr, bbox, size_map, offset_map = self.box_head(opt_feat, gt_score_map)
            # outputs_coord = box_xyxy_to_cxcywh(bbox)
            outputs_coord = bbox
            outputs_coord_new = outputs_coord.view(bs, Nq, 4)
            out = {'pred_boxes': outputs_coord_new,
                   'score_map': score_map_ctr,
                   'size_map': size_map,
                   'offset_map': offset_map}
            return out
        else:
            raise NotImplementedError


def build_ostrack(cfg, training=True):
    current_dir = os.path.dirname(os.path.abspath(__file__))  # This is your Project Root
    pretrained_path = os.path.join(current_dir, '../../../pretrained')
    if cfg.MODEL.PRETRAIN_FILE and ('OSTrack' not in cfg.MODEL.PRETRAIN_FILE) and training:
        pretrained = os.path.join(pretrained_path, cfg.MODEL.PRETRAIN_FILE)
    else:
        pretrained = ''

    if cfg.MODEL.BACKBONE.TYPE == 'vit_base_patch16_224':
        backbone = vit_base_patch16_224(pretrained, drop_path_rate=cfg.TRAIN.DROP_PATH_RATE)
        hidden_dim = backbone.embed_dim
        patch_start_index = 1

    elif cfg.MODEL.BACKBONE.TYPE == 'vit_base_patch16_224_ce':
        backbone = vit_base_patch16_224_ce(pretrained, drop_path_rate=cfg.TRAIN.DROP_PATH_RATE,
                                           ce_loc=cfg.MODEL.BACKBONE.CE_LOC,
                                           ce_keep_ratio=cfg.MODEL.BACKBONE.CE_KEEP_RATIO,
                                           )
        hidden_dim = backbone.embed_dim
        patch_start_index = 1

    elif cfg.MODEL.BACKBONE.TYPE == 'vit_large_patch16_224_ce':
        backbone = vit_large_patch16_224_ce(pretrained, drop_path_rate=cfg.TRAIN.DROP_PATH_RATE,
                                            ce_loc=cfg.MODEL.BACKBONE.CE_LOC,
                                            ce_keep_ratio=cfg.MODEL.BACKBONE.CE_KEEP_RATIO,
                                            )

        hidden_dim = backbone.embed_dim
        patch_start_index = 1

    else:
        raise NotImplementedError

    backbone.finetune_track(cfg=cfg, patch_start_index=patch_start_index)
    language_backbone = build_bert()
    box_head = build_box_head(cfg, hidden_dim)

    model = OSTrack(
        backbone,
        language_backbone,
        box_head,
        aux_loss=False,
        head_type=cfg.MODEL.HEAD.TYPE,
    )

    if 'OSTrack' in cfg.MODEL.PRETRAIN_FILE and training:
        checkpoint = torch.load(cfg.MODEL.PRETRAIN_FILE, map_location="cpu")
        missing_keys, unexpected_keys = model.load_state_dict(checkpoint["net"], strict=False)
        print('missing_keys')
        print(missing_keys)
        print('unexpercted_keys')
        print(unexpected_keys)
        print('Load pretrained model from: ' + cfg.MODEL.PRETRAIN_FILE)

    return model
