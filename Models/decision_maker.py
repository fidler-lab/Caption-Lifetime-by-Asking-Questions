import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ConvBlock(nn.Module):
    def __init__(self, ic, oc, ks, dropout=True):
        super(ConvBlock, self).__init__()

        self.dropout = dropout
        self.conv = torch.nn.Conv1d(in_channels=ic, out_channels=oc, kernel_size=ks)
        self.nl = nn.PReLU()

    def forward(self, input):

        output = self.conv(input)
        output = self.nl(output)
        if self.dropout:
            output = torch.dropout(output, p=0.5, train=True)
        return output


class FCBlock(nn.Module):
    def __init__(self, isize, osize, dropout=True):
        super(FCBlock, self).__init__()

        self.dropout = dropout
        self.fc = nn.Linear(isize, osize)
        self.nl = nn.PReLU()

    def forward(self, input):

        output = self.fc(input)
        output = self.nl(output)
        if self.dropout:
            output = torch.dropout(output, p=0.5, train=True)
        return output


class Base(nn.Module):
    def __init__(self, opt):
        super(Base, self).__init__()

        self.k = opt.k
        self.pos_vocab_size = opt.pos_vocab_size
        self.zero_mat = torch.zeros([opt.batch_size, opt.c_max_sentence_len + 1, self.k], device=device)
        self.eye_mat = torch.eye(n=self.k, m=self.k, device=device)

    def min_distance(self, word_embedding, word_idxs):
        emb = word_embedding[word_idxs]

        large_number = 1e6
        num_words = emb.size(0)

        pairwise_differences = emb.unsqueeze(1).repeat(1, num_words, 1) - emb.unsqueeze(0)
        pairwise_distances = torch.norm(pairwise_differences, p=2, dim=2) + large_number * self.eye_mat[:num_words, :num_words].clone()
        return pairwise_distances.min(dim=0)[0]

    def cosine_distance_to_others(self, word_embedding, word_idxs):
        emb = word_embedding[word_idxs]

        cos_dist = F.cosine_similarity(emb[0].unsqueeze(0), emb, dim=1)
        cos_dist[0] = 0.0  # pad similarity against self to be 0 instead of 1. will be better for conv filters

        return cos_dist

    def cosine_distance_sentence(self, word_embedding, word_idxs, sentence):
        sentence_emb = word_embedding[sentence].sum(dim=0)
        emb = word_embedding[word_idxs]

        return F.cosine_similarity(sentence_emb.unsqueeze(0), emb, dim=1)

    def compute_uncertainty_features(self, topk, caption, cap_len, cap_embedding):
        batch_size = topk[0][0].size(0)

        min_dist = self.zero_mat[:batch_size].clone()
        cos_dist = self.zero_mat[:batch_size].clone()
        sentence_cos_dist = self.zero_mat[:batch_size].clone()

        for t in range(len(topk)):
            probs, words = topk[t]
            probs, words = probs, words
            for i in range(words.size(0)):
                sentence_cos_dist[i, t] = self.cosine_distance_sentence(cap_embedding, words[i], caption[i][:cap_len[i]])
                min_dist[i, t] = self.min_distance(cap_embedding, words[i])
                cos_dist[i, t] = self.cosine_distance_to_others(cap_embedding, words[i])

        return min_dist, cos_dist, sentence_cos_dist

    def valid_pos(self, pos):
        return torch.max(pos, dim=2)[1] != self.pos_vocab_size-1


class DecisionMaker(Base):

    def __init__(self, opt):
        super(DecisionMaker, self).__init__(opt)

        self.c_vocab_size = opt.c_vocab_size
        self.dropout = opt.use_dropout

        self.dm_emb_size = opt.dm_emb_size
        self.pos_emb_size = opt.pos_emb_size

        self.dm_rnn_size = opt.dm_rnn_size
        self.dm_cnn_size = opt.dm_cnn_size
        self.use_caption_hidden = opt.use_caption_hidden

        # embedding matrices
        self.caption_embedding = nn.Embedding(self.c_vocab_size + 1, self.dm_emb_size, padding_idx=self.c_vocab_size)
        self.pos_embedding = nn.Embedding(self.pos_vocab_size+1, self.pos_emb_size, padding_idx=self.pos_vocab_size)

        self.caption_encoder = nn.GRU(self.dm_emb_size, self.dm_rnn_size, 1, batch_first=True, dropout=0.5, bidirectional=True)

        # Conv layers
        layers = []
        layers.append(ConvBlock(ic=4, oc=self.dm_cnn_size / 4, ks=3, dropout=self.dropout))
        layers.append(ConvBlock(ic=self.dm_cnn_size / 4, oc=self.dm_cnn_size / 2, ks=3, dropout=self.dropout))
        layers.append(ConvBlock(ic=self.dm_cnn_size / 2, oc=self.dm_cnn_size, ks=2, dropout=self.dropout))
        self.CNN = nn.Sequential(*layers)

        # FC layers
        sizeP, sizeH, sizeI, sizeC = [256, 256, 512, 512]
        self.pBlock = FCBlock(isize=self.pos_emb_size, osize=sizeP, dropout=self.dropout)
        self.iBlock = FCBlock(isize=opt.image_channels, osize=sizeI, dropout=self.dropout)
        self.fcO = nn.Linear(self.dm_cnn_size + sizeC, 1)

        if self.use_caption_hidden:
            self.hBlock = FCBlock(isize=opt.c_rnn_size, osize=sizeH, dropout=self.dropout)
            self.cBlock = FCBlock(isize=sizeH+sizeP+sizeI+2*self.dm_rnn_size, osize=sizeC, dropout=self.dropout)
        else:
            self.cBlock = FCBlock(isize=sizeP+sizeI+2*self.dm_rnn_size, osize=sizeC, dropout=self.dropout)

    def forward(self, hidden, attended_img, caption, cap_len, pos, topk, cap_embedding):

        valid_pos = self.valid_pos(pos)

        min_dist, cos_dist, sentence_cos_dist = self.compute_uncertainty_features(
            topk, caption, cap_len, cap_embedding)

        # uncertainty features
        # probabilities of top k words
        probs = torch.cat([topk[t][0].unsqueeze(1) for t in range(len(topk))], dim=1)
        uncertainty_features = torch.cat([x.unsqueeze(2) for x in [min_dist, cos_dist, sentence_cos_dist, probs]], dim=2)
        batch_size, caption_len, channels, k = uncertainty_features.size()
        uncertainty_features = uncertainty_features.contiguous().view(-1, channels, k)

        x = self.CNN(uncertainty_features)
        unc_feat = x.view(batch_size, caption_len, self.dm_cnn_size)

        # context features: caption
        cap_emb = self.caption_embedding(caption)
        _, cap_feat = self.caption_encoder(cap_emb)
        cap_feat = torch.cat([cap_feat[0], cap_feat[1]], dim=1).unsqueeze(1).repeat(1, caption_len, 1)

        pos_emb = torch.matmul(pos, self.pos_embedding.weight)
        pos_feat = self.pBlock(pos_emb)

        # context features: image
        img_feat = self.iBlock(attended_img)

        # fuse (caption, pos, image, hidden) features
        if self.use_caption_hidden:
            hid_feat = self.hBlock(hidden)
            context_feat = torch.cat((cap_feat, pos_feat, img_feat, hid_feat), dim=2)
        else:
            context_feat = torch.cat((cap_feat, pos_feat, img_feat), dim=2)

        context_feat = self.cBlock(context_feat)

        # fuse (context, uncertainty) features
        fuse_feat = torch.cat((context_feat, unc_feat), dim=2)
        logits = self.fcO(fuse_feat).squeeze(2)

        return logits, valid_pos
