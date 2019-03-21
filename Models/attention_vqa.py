import torch
import torch.nn as nn
from argparse import Namespace


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class AttentionI(nn.Module):

    def __init__(self, opt):

        super(AttentionI, self).__init__()

        self.att_hidden_size = opt.att_hidden_size
        self.rnn_size = opt.rnn_size
        self.img_channels = opt.image_channels
        self.img_width = opt.image_feature_size
        self.img_area = self.img_width * self.img_width

        self.fc1 = nn.Sequential(
            nn.Linear(self.img_channels + self.rnn_size, self.att_hidden_size),
            nn.PReLU(),
            nn.Dropout(opt.dropout)
        )
        self.fc2 = nn.Linear(self.att_hidden_size, 1)

    def forward(self, img_features, hidden):

        z = hidden.unsqueeze(1).repeat(1, self.img_area, 1)  # Nx49x512

        z = torch.cat((img_features, z), 2)  # Nx49x2560
        z = self.fc1(z)
        z = self.fc2(z)  # Nx49x1
        a = torch.softmax(z, dim=1)

        img_emb = (img_features * a).sum(1)

        return img_emb, a


class AttentionC(nn.Module):

    def __init__(self, opt):

        super(AttentionC, self).__init__()

        self.att_hidden_size = opt.att_hidden_size
        self.rnn_size = opt.rnn_size
        self.img_channels = opt.image_channels
        self.img_width = opt.image_feature_size
        self.img_area = self.img_width * self.img_width

        self.zero_vec = torch.zeros([opt.batch_size, 1, self.att_hidden_size], requires_grad=False).to(device)

        self.fc1 = nn.Sequential(
            nn.Linear(self.rnn_size + self.rnn_size, self.att_hidden_size),
            nn.PReLU(),
            nn.Dropout(opt.dropout)
        )
        self.fc2 = nn.Linear(self.att_hidden_size, 1)
        self.nonlin = nn.PReLU()
        self.dropout = nn.Dropout(opt.dropout)

    def forward(self, c_hidden, q_hidden, batch_size):

        c_hidden = c_hidden.squeeze()
        c_hidden = c_hidden.view(batch_size, -1, c_hidden.size(1))  # Nx5x512

        # introduce a zero vector so we can choose to not attend on captions
        c_hidden = torch.cat((c_hidden, self.zero_vec[:batch_size]), 1)
        q_hidden = q_hidden.unsqueeze(1).repeat(1, c_hidden.size(1), 1)  # Nx6x512

        z = torch.cat((c_hidden, q_hidden), 2)  # Nx6x1024
        z = self.fc1(z)
        z = self.fc2(z)  # Nx6x1
        a = torch.softmax(z, dim=1)  # Nx6x1

        cap_emb = (c_hidden * a).sum(1)

        return cap_emb, a


class AttentionVQA(nn.Module):

    def __init__(self, opt):
        super(AttentionVQA, self).__init__()

        self.q_vocab_size = opt.q_vocab_size
        self.a_vocab_size = opt.a_vocab_size
        self.c_vocab_size = opt.c_vocab_size
        self.rnn_layers = opt.rnn_layers
        self.rnn_size = opt.rnn_size

        self.q_emb_size = opt.word_embedding_size
        self.img_channels = opt.image_channels
        self.img_width = opt.image_feature_size
        self.img_area = self.img_width * self.img_width

        # Embedding layers
        self.question_embedding = nn.Embedding(self.q_vocab_size+1, self.q_emb_size, padding_idx=self.q_vocab_size)
        self.caption_embedding = nn.Embedding(self.c_vocab_size+1, self.q_emb_size, padding_idx=self.c_vocab_size)

        # Decoders
        self.qgru = nn.GRU(self.q_emb_size, self.rnn_size, self.rnn_layers, batch_first=True, dropout=opt.dropout)
        self.cgru = nn.GRU(self.q_emb_size, self.rnn_size, self.rnn_layers, batch_first=True, dropout=opt.dropout)

        self.bnh = nn.BatchNorm1d(self.rnn_size)
        self.bn0 = nn.BatchNorm1d(self.img_channels)
        self.img_attention = AttentionI(opt)
        self.cap_attention = AttentionC(opt)

        self.fc1 = nn.Sequential(
            nn.Linear(self.rnn_size, self.rnn_size),
            nn.BatchNorm1d(self.rnn_size),
            nn.PReLU()
        )

        self.fc2 = nn.Sequential(
            nn.Linear(self.img_channels, self.rnn_size),
            nn.BatchNorm1d(self.rnn_size),
            nn.PReLU()
        )

        self.fc3 = nn.Sequential(
            nn.Linear(self.rnn_size, self.rnn_size),
            nn.BatchNorm1d(self.rnn_size),
            nn.PReLU()
        )

        self.fc4 = nn.Sequential(
            nn.Linear(self.rnn_size + self.rnn_size, self.img_channels),
            nn.BatchNorm1d(self.img_channels),
            nn.PReLU()
        )

        self.dropout = nn.Dropout(opt.dropout)
        self.fc5 = nn.Linear(self.img_channels, self.a_vocab_size)

        initrange = 0.1
        self.question_embedding.weight.data[:-1].uniform_(-initrange, initrange)  # -1 to deal with pad index
        self.caption_embedding.weight.data[:-1].uniform_(-initrange, initrange)

    def forward(self, img_features, question, captions):
        batch_size = question.size(0)

        # encode question
        q_emb = self.question_embedding(question)
        _, last_hidden = self.qgru(q_emb)
        last_hidden = self.bnh(last_hidden.squeeze())  # Nx512

        # encode and attend image
        img_features = self.normalize_img(img_features)
        img_emb, a_img = self.img_attention(img_features, last_hidden)

        # encode and attend captions
        captions = captions.view(-1, captions.size(2))  # (N*5)x14
        c_emb = self.caption_embedding(captions)  # (N*5)x14x300
        _, c_last_hidden = self.cgru(c_emb)  # (N*5)x512

        cap_emb, a_cap = self.cap_attention(c_last_hidden, last_hidden, batch_size)

        # combine embeddings
        q = self.fc1(last_hidden)
        i = self.fc2(img_emb)
        c = self.fc3(cap_emb)
        h1 = q*i
        h2 = q*c

        # multilayer perceptron
        h = torch.cat((h1, h2), 1)  # Nx1024
        h = self.fc4(self.dropout(h))
        logits = self.fc5(h)

        return Namespace(probs=torch.softmax(logits, dim=1), logits=logits, a_img=a_img, a_cap=a_cap)

    def normalize_img(self, img_features):
        # normalize image features
        img_features = img_features.contiguous().view(-1, self.img_channels, self.img_area)  # Nx2048x49
        img_features = self.bn0(img_features)  # I feel like batch norm is better than L2 normalization
        img_features = img_features.permute(0, 2, 1)  # Nx49x2048

        return img_features