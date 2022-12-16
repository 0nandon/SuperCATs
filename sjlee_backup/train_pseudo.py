
"""
1. config 아래와 같이 설정
2. weights은 상황에 맞게 indoor, outdoor 설정해주어야 함
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

"""
# start training
for epoch in range(1, opt.epoch+1):
    epoch_loss = 0
    superglue.double().train()
    for i, pred in enumerate(train_loader):
        for k in pred:
            if k != 'file_name' and k!='image0' and k!='image1':
                if type(pred[k]) == torch.Tensor:
                    pred[k] = Variable(pred[k].cuda())
                else:
                    pred[k] = Variable(torch.stack(pred[k]).cuda())
        
        # =========== new code =============== #
        scores, data = superglue(pred)
        loss = loss_superglue(scores, data['all_matches'].permute(1, 2, 0))

        for k, v in pred.items():
            pred[k] = v[0]
        pred = {**pred, **data, **{'loss', loss}}

         # ... keep going
"""