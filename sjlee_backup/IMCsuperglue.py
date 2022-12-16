
import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
from functools import partial

from pydoc import source_synopsis
from sjlee_backup.superglue2 import SuperGlue, normalize_keypoints, arange_like, log_optimal_transport
from sjlee_backup.losssuperglue import loss_superglue

sys.path.append(os.path.join(os.path.dirname(__file__), 'cats'))
from cats import TransformerAggregator

def dfs_freeze(model):
    for name, child in model.named_children():
        for param in child.parameters():
            param.requires_grad = False

        dfs_freeze(child)

def softmax_with_temperature(x, beta=2., d = 1):
    r'''SFNet: Learning Object-aware Semantic Flow (Lee et al.)'''
    M, _ = x.max(dim=d, keepdim=True)
    x = x - M # subtract maximum value for stability
    exp_x = torch.exp(x/beta)
    exp_x_sum = exp_x.sum(dim=d, keepdim=True)
    return exp_x / exp_x_sum

# positional embedding 필요한가?
# M * N 크기가 다 다른 문제
class SimpleSuperCATs(SuperGlue):
    def __init__(self, 
    config,
    feature_size=32,
    feature_proj_dim=128,
    depth=4,
    num_heads=4,
    mlp_ratio=4,
    ):
        super().__init__(config)

        # freeze superglue's layers
        dfs_freeze(self.kenc)
        dfs_freeze(self.gnn)
        dfs_freeze(self.final_proj)

        self.feature_size = feature_size
        self.feature_proj_dim = feature_proj_dim
        self.decoder_embed_dim = self.feature_size ** 2

        self.decoder = TransformerAggregator(
            img_size=self.feature_size, embed_dim=self.decoder_embed_dim, depth=depth, num_heads=num_heads,
            mlp_ratio=mlp_ratio, qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6),
            num_hyperpixel=1
        )

    def forward(self, data):
        """Run SuperGlue on a pair of keypoints and descriptors"""
        with torch.no_grad():
        
            desc0, desc1 = data['descriptors0'], data['descriptors1']
            kpts0, kpts1 = data['keypoints0'], data['keypoints1']

        

            desc0 = desc0.transpose(0,1)
            desc1 = desc1.transpose(0,1)
            kpts0 = torch.reshape(kpts0, (1, -1, 2))
            kpts1 = torch.reshape(kpts1, (1, -1, 2))
        
            if kpts0.shape[1] == 0 or kpts1.shape[1] == 0:  # no keypoints
                shape0, shape1 = kpts0.shape[:-1], kpts1.shape[:-1]
                return [], {
                    'matches0': kpts0.new_full(shape0, -1, dtype=torch.int)[0],
                    'matches1': kpts1.new_full(shape1, -1, dtype=torch.int)[0],
                    'matching_scores0': kpts0.new_zeros(shape0)[0],
                    'matching_scores1': kpts1.new_zeros(shape1)[0],
                    'skip_train': True
                }
            
            # Keypoint normalization.
            kpts0 = normalize_keypoints(kpts0, data['image0'].shape)
            kpts1 = normalize_keypoints(kpts1, data['image1'].shape)
            
            # Keypoint MLP encoder.
            desc0 = desc0 + self.kenc(kpts0, torch.transpose(data['scores0'], 0, 1))
            desc1 = desc1 + self.kenc(kpts1, torch.transpose(data['scores1'], 0, 1))

            # Multi-layer Transformer network.
            desc0, desc1 = self.gnn(desc0, desc1)

            # Final MLP projection.
            mdesc0, mdesc1 = self.final_proj(desc0), self.final_proj(desc1)

            # Compute matching descriptor distance.
            scores = torch.einsum('bdn,bdm->bnm', mdesc0, mdesc1)
            scores = scores / self.config['descriptor_dim']**.5
        
        #print(scores.max(), scores.min())
        
        # Run the optimal transport.
        
        scores = log_optimal_transport(
            scores, self.bin_score,
            iters=self.config['sinkhorn_iterations'])     
        
        # Get the matches with score above "match_threshold".
        max0, max1 = scores[:, :-1, :-1].max(2), scores[:, :-1, :-1].max(1)
        indices0, indices1 = max0.indices, max1.indices
        mutual0 = arange_like(indices0, 1)[None] == indices1.gather(1, indices0)
        mutual1 = arange_like(indices1 , 1)[None] == indices0.gather(1, indices1)
        zero = scores.new_tensor(0)
        mscores0 = torch.where(mutual0, max0.values.exp(), zero)
        mscores1 = torch.where(mutual1, mscores0.gather(1, indices1), zero)
        valid0 = mutual0 & (mscores0 > self.config['match_threshold'])
        valid1 = mutual1 & valid0.gather(1, indices1)
        indices0 = torch.where(valid0, indices0, indices0.new_tensor(-1))
        indices1 = torch.where(valid1, indices1, indices1.new_tensor(-1))

        #print(mscores0.min(), mscores0.max())
        #print(mscores0)

        return scores, {
            'matches0': indices0[0], # use -1 for invalid match
            'matches1': indices1[0], # use -1 for invalid match
            'matching_scores0': mscores0[0],
            'matching_scores1': mscores1[0],
            'skip_train': False
        }


if __name__ == '__main__':
    from superpoint import SuperPoint

    config = {
        'superpoint': {
            'nms_radius': 4,
            'keypoint_threshold': 0.005,
            'max_keypoints': 1024
        },
        'superglue': {
            'weights': 'outdoor',
            'sinkhorn_iterations': 20,
            'match_threshold':0.2
        }
    }

    """
    data = {
        'image0': torch.randn(1, 1, 512, 512),
        'image1': torch.randn(1, 1, 512, 512)
    }

    superpoint = SuperPoint(config.get('superpoint', {}))

    output1 = superpoint({'image': data['image0']})
    output2 = superpoint({'image': data['image1']})

    pred = {}

    pred = {**pred, **{k+'0': v for k, v in output1.items()}}
    pred = {**pred, **{k+'1': v for k, v in output2.items()}}

    data = {**data, **pred}

    for k in data:
        if isinstance(data[k], (list, tuple)):
            data[k] = torch.stack(data[k])
    """

    pred = {
        'keypoints0' : torch.randn(1, 1, 484, 2),
        'keypoints1' : torch.randn(1, 1, 484, 2),
        'descriptors0' : torch.randn(256, 1, 484),
        'descriptors1' : torch.randn(256, 1, 484),
        'scores0' : torch.randn(484, 1),
        'scores1' : torch.randn(484, 1),
        'image0' : torch.randn(1, 1, 512, 512),
        'image1' : torch.randn(1, 1, 512, 512),
        # 'all_matches' : torch.randn(2, 1, 1248)
    }

    superglue = SimpleSuperCATs(config.get('superglue', {}))
    scores, output = superglue(pred)

    # loss = loss_superglue(scores, pred['all_matches'].permute(1, 2, 0))
    # print(loss)