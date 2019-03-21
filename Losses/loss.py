import torch
from torch.autograd import Variable


def _sequence_mask(sequence_length, max_len=None):
    if max_len is None:
        max_len = sequence_length.data.max()
    batch_size = sequence_length.size(0)
    seq_range = torch.arange(0, max_len).long()
    seq_range_expand = seq_range.unsqueeze(0).expand(batch_size, max_len)
    seq_range_expand = Variable(seq_range_expand)
    if sequence_length.is_cuda:
        seq_range_expand = seq_range_expand.cuda()
    seq_length_expand = (sequence_length.unsqueeze(1)
                         .expand_as(seq_range_expand))
    return seq_range_expand < seq_length_expand


# Cross-entropy loss masked by sequence length
def masked_CE(logits, target, length, weights=None):
    logits_flat = logits.view(-1, logits.size(-1))  # logits_flat: (batch * max_len, num_classes)
    log_probs_flat = torch.log_softmax(logits_flat, dim=1)  # log_probs_flat: (batch * max_len, num_classes)
    target_flat = target.view(-1, 1)  # target_flat: (batch * max_len, 1)
    losses_flat = -torch.gather(log_probs_flat, dim=1, index=target_flat)  # losses_flat: (batch * max_len, 1)
    losses = losses_flat.view(*target.size())  # losses: (batch, max_len)
    mask = _sequence_mask(sequence_length=length, max_len=target.size(1))  # mask: (batch, max_len)
    losses = losses * mask.float()
    if weights is not None:
        losses = weights.unsqueeze(1) * losses
    loss = losses.sum() / length.float().sum()
    return loss


def seq_max_and_mask(logits, length, max_len):
    mask = _sequence_mask(length, max_len)
    _, pred = torch.max(logits, dim=2)

    return pred*mask.long()


# Policy gradient loss masked
def masked_PG(reward, log_probs, mask, eps=0.0001):
    mask = mask.float()
    loss = -reward*log_probs*mask
    loss = torch.sum(loss)/(torch.sum(mask) + eps)
    return loss