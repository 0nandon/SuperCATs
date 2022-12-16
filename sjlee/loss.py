import torch

def loss_superglue(scores, all_matches):
    # check if indexed correctly
    loss = []
    for i in range(len(all_matches[0])):
        x = all_matches[0][i][0]
        y = all_matches[0][i][1]
        loss.append(-torch.log( scores[0][x][y].exp() + 1e-7 )) # check batch size == 1 ?
    # for p0 in unmatched0:
    #     loss += -torch.log(scores[0][p0][-1])
    # for p1 in unmatched1:
    #     loss += -torch.log(scores[0][-1][p1])
    loss_mean = torch.mean(torch.stack(loss))
    loss_mean = torch.reshape(loss_mean, (1, -1))
    return loss_mean[0]
