# Third Party
import torch
import torch.nn as nn

# For intuition, we will have a trajectory and our batch will be a sequence of different subsequent steps in the trajectory
# So we will have (BEV_g, BEV_a) for each frame we can then efficiently use npairs/lifed structure to try out all combinations in the minibatch
def compute_local_traj_loss(embed_g, embed_a, labels, loss_fn):
    loss_batch = 0.0
    embeddings_all = torch.zeros((embed_g.shape[0]+1, embed_g.shape[1]), device = embed_g.device)
    for i in range(labels.shape[0]):
        labels_batch = labels[i]
        # Extract positive and negative idxs
        pos_idx = torch.argwhere(labels_batch==1)
        neg_idx = torch.argwhere(labels_batch==0)

        # Create pairs
        anchors_pos = torch.ones_like(pos_idx) * len(labels_batch)
        anchors_neg = torch.ones_like(neg_idx) * len(labels_batch)
        
        # Calculate the loss for the pairs, let the particular loss sort out the details
        embeddings_all[:-1] = embed_a.clone()
        embeddings_all[-1]  = embed_g[i].clone()
        loss = loss_fn.forward(embeddings_all.clone(), indices_tuple=(anchors_pos, pos_idx, anchors_neg, neg_idx))
        
        # Backprop and optimize
        # if i < labels.shape[0]-1:
        #     self.backward(loss)
        # else:
        #     loss.backward(retain_graph=False)

        # Accumulate loss
        loss_batch += loss

    return loss_batch

class ContrastiveCosineLoss(nn.Module):
    def __init__(self, m = 0.5):
        super().__init__()
        self.m = m

    def forward(self, embed_g, embed_a, pos, neg):
        """
        Naive Cosine Loss using a Margin.
        We are going to compare the ground feature embeddings to all the aerial feature embeddings.
        Those that are close enough are positives.
        Those far enough away are negatives and the loss will be reflected to push them further with the margin
        """

        # Flatten tensors
        neg = neg.flatten()
        pos = pos.flatten()

        # Compute sim and loss
        sim = (embed_g @ embed_a.T).flatten()
        dist = torch.sqrt(2*(1-sim))
        loss_pos = dist[pos]
        loss_neg = torch.clip(self.m - dist[neg], min = 0)

        mean_loss_neg = 0.
        mean_loss_pos = 0.
        if len(loss_neg) > 0:
            mean_loss_neg = torch.mean(loss_neg)
        if len(loss_pos) > 0:
            mean_loss_pos = torch.mean(loss_pos)

        return mean_loss_pos + mean_loss_neg
    
class ContrastiveCosineGaussianLoss(torch.nn.Module):
    def __init__(self, m = 1, sigma=32.0):
        super(ContrastiveCosineGaussianLoss, self).__init__()
        self.sigma = sigma
        self.m     = m

    def forward(self, embed_g, embed_a, pos, neg, physical_dists):
        # Flatten tensors
        neg = torch.tensor(neg.flatten())
        pos = torch.tensor(pos.flatten())

        # Compute cosine similarity
        sim  = (embed_g @ embed_a.T).flatten()
        dist = torch.sqrt(2*(1-sim))

        # Define contrastive loss and gaussian penalty loss
        import numpy as np
        if type(physical_dists) == np.ndarray:
            physical_dists = torch.tensor(physical_dists)
        gaussian_w = torch.exp(-0.5 * (physical_dists/self.sigma)**2)
        gaussian_w = gaussian_w.to(dist.device)
        loss_pos = dist[pos] * gaussian_w[pos]
        # print(torch.mean(loss_pos))
        loss_neg = torch.clip(self.m - dist[neg], min = 0)
        # print(torch.mean(loss_neg))

        return torch.mean(loss_pos) + torch.mean(loss_neg)