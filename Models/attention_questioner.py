import torch
import torch.nn as nn
from argparse import Namespace


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Attention(nn.Module):

    def __init__(self, opt):

        super(Attention, self).__init__()

        self.att_hidden_size = opt.att_hidden_size
        self.rnn_hidden_size = opt.rnn_size

        self.img_channels = opt.image_channels
        self.img_width = opt.image_feature_size
        self.img_area = self.img_width * self.img_width

        self.fc1 = nn.Linear(self.rnn_hidden_size, self.att_hidden_size)
        self.fc2 = nn.Linear(self.att_hidden_size, 1)
        self.fc3 = nn.Linear(2*self.img_channels, self.img_channels)

    def forward(self, img_features, att_img_features, hidden, a_cap):

        # self attention
        z = self.fc1(hidden)
        z = torch.tanh(att_img_features + z.permute(1, 0, 2).repeat(1, self.img_area, 1))
        z = self.fc2(z)
        a = torch.softmax(z, dim=1)
        self_features = (img_features * a).sum(1)

        # captioner attention
        cap_features = (img_features * a_cap.unsqueeze(2)).sum(1)

        # concatenate self image feature and captioner's
        img_emb = torch.cat((self_features, cap_features), dim=1)

        # some FC layers on image features
        img_emb = torch.relu(self.fc3(img_emb))

        return img_emb, a


class Decoder(nn.Module):

    def __init__(self, opt):

        super(Decoder, self).__init__()

        self.q_vocab_size = opt.q_vocab_size

        self.word_emb_size = opt.word_embedding_size
        self.pos_emb_size = opt.pos_emb_size
        self.cap_rnn_size = opt.cap_rnn_size

        self.rnn_size = opt.rnn_size
        self.rnn_layers = opt.rnn_layers
        self.img_channels = opt.image_channels

        self.dropout_rate = opt.dropout

        self.gru = nn.GRU(2 * self.cap_rnn_size + self.word_emb_size + self.img_channels, self.rnn_size, self.rnn_layers,
                          batch_first=True, dropout=self.dropout_rate)

        self.dropout = nn.Dropout(self.dropout_rate)
        self.fc1 = nn.Linear(self.rnn_size, self.q_vocab_size + 1)

    def forward(self, hidden, img_emb, c_emb, q_emb):

        x = torch.cat((img_emb, c_emb, q_emb), 1)

        hidden, last_hidden = self.gru(x.unsqueeze(1), hidden)

        logits = self.fc1(self.dropout(hidden.squeeze()))

        return logits, last_hidden


class QuestionGenerator(nn.Module):

    def __init__(self, opt):

        super(QuestionGenerator, self).__init__()

        self.q_vocab_size = opt.q_vocab_size
        self.c_vocab_size = opt.c_vocab_size
        self.p_vocab_size = opt.p_vocab_size
        self.special_symbols = opt.special_symbols

        self.att_hidden_size = opt.att_hidden_size
        self.rnn_layers = opt.rnn_layers
        self.use_position_encoding = opt.use_position_encoding
        self.use_caption_hidden = opt.use_caption_hidden

        self.word_emb_size = opt.word_embedding_size
        self.pos_emb_size = opt.pos_emb_size
        self.rnn_size = opt.rnn_size
        self.cap_rnn_size = opt.cap_rnn_size
        self.cap_hidden_size = opt.cap_hidden_size

        self.img_channels = opt.image_channels
        self.img_width = opt.image_feature_size
        self.img_area = self.img_width * self.img_width
        self.ss_prob = 0.0
        self.dropout_rate = opt.dropout

        # dummy vectors
        self.one_vec = torch.ones([3*opt.batch_size, 1], dtype=torch.long, device=device)

        # submodules
        self.attention = Attention(opt)
        self.decoder = Decoder(opt)

        if self.use_position_encoding:
            input_size = 2*self.cap_rnn_size
        else:
            input_size = self.cap_rnn_size

        self.caption_encoder = nn.GRU(input_size, self.cap_rnn_size, self.rnn_layers, batch_first=True, dropout=self.dropout_rate, bidirectional=True)

        # embedding matrices
        self.question_embedding = nn.Embedding(self.q_vocab_size+1, self.word_emb_size, padding_idx=self.q_vocab_size)
        self.caption_embedding = nn.Embedding(self.c_vocab_size+1, self.cap_rnn_size, padding_idx=self.c_vocab_size)
        self.pos_embedding = nn.Embedding(self.p_vocab_size+1, self.pos_emb_size, padding_idx=self.p_vocab_size)
        self.context_embedding = nn.Linear(self.cap_hidden_size, self.word_emb_size)

        self.fc1 = nn.Linear(self.img_channels, self.att_hidden_size)

        if self.use_caption_hidden:
            input_size = self.word_emb_size + self.pos_emb_size
        else:
            input_size = self.pos_emb_size

        self.fc2 = nn.Linear(input_size, self.rnn_size)

        initrange = 0.1
        self.question_embedding.weight.data[:-1].uniform_(-initrange, initrange)  # -1 to deal with pad index

    def forward(self, img_features, caption, pos, context, att_weights, source, q_idx_vec):
        batch_size, seq_length = img_features.size(0), source.size(1)

        # encode the caption
        if self.use_position_encoding:
            input = torch.cat([self.caption_embedding(caption), q_idx_vec], dim=2)  # q_idx is position encoding vector
        else:
            input = self.caption_embedding(caption)

        _, cap_emb = self.caption_encoder(input)
        cap_emb = torch.cat([cap_emb[0], cap_emb[1]], dim=1)

        # initialize the first hidden state of the RNN decoder
        context_emb = torch.relu(self.context_embedding(context))
        pos_emb = torch.relu(torch.matmul(pos, self.pos_embedding.weight))

        if self.use_caption_hidden:
            input = torch.cat([context_emb, pos_emb], 1)
        else:
            input = pos_emb

        hidden = self.fc2(input).unsqueeze(0)

        # permute img dimensions and downsize
        img_features = self.vectorize_img(img_features)

        logits_list = []

        for i in range(seq_length):

            # scheduled sampling
            if i >= 1 and self.ss_prob > 0.0:
                sample_prob = torch.Tensor(batch_size).uniform_(0, 1).to(device)
                sample_mask = sample_prob < self.ss_prob
                if sample_mask.sum().item() == 0:
                    previous_word = source[:, i]
                else:
                    sample_ind = sample_mask.nonzero().view(-1)
                    previous_word = source[:, i].clone()

                    prob_prev = torch.softmax(logits, dim=1)  # fetch prev distribution: shape batchsize x vocab
                    previous_word.index_copy_(0, sample_ind, torch.multinomial(prob_prev, 1).view(-1).index_select(0, sample_ind))
                    previous_word.requires_grad_(False)
            else:
                previous_word = source[:, i]

            # compute attended image feature and word embeddings
            img_emb, _ = self.attention(img_features, self.fc1(img_features), hidden, att_weights)
            q_emb = self.question_embedding(previous_word)

            logits, hidden = self.decoder(hidden, img_emb, cap_emb, q_emb)

            logits_list.append(logits.unsqueeze(1))

        return torch.cat(logits_list, dim=1)

    def sample(self, img_features, caption, pos, context, att_weights, q_idx_vec, greedy=False, max_seq_len=15, temperature=1.0, hidden=None, previous_word=None):
        batch_size = img_features.size(0)

        # encode the caption
        if self.use_position_encoding:
            input = torch.cat([self.caption_embedding(caption), q_idx_vec], dim=2)  # q_idx is position encoding vector
        else:
            input = self.caption_embedding(caption)

        _, cap_emb = self.caption_encoder(input)
        cap_emb = torch.cat([cap_emb[0], cap_emb[1]], dim=1)

        # initialize the first hidden state of the RNN decoder
        if hidden is None:
            context_emb = torch.relu(self.context_embedding(context))
            pos_emb = torch.relu(torch.matmul(pos, self.pos_embedding.weight))

            if self.use_caption_hidden:
                input = torch.cat([context_emb, pos_emb], 1)
            else:
                input = pos_emb

            hidden = self.fc2(input).unsqueeze(0)

        img_features = self.vectorize_img(img_features)
        att_img_features = self.fc1(img_features)

        question, log_probs, mask, hidden_arr, attention_arr = [], [], [], [], []

        if previous_word is None:
            previous_word = self.one_vec[:batch_size]  # beginning of sentence token is 1

        for i in range(max_seq_len):

            # compute attended image feature and word embeddings
            img_emb, attention = self.attention(img_features, att_img_features, hidden, att_weights)
            q_emb = self.question_embedding(previous_word).squeeze()

            # roll out decoder
            logits, hidden = self.decoder(hidden, img_emb, cap_emb, q_emb)

            log_prob = torch.log_softmax(logits, dim=1)
            # sample: greedy or wrt some temperature
            if greedy:
                log_prob, word = torch.max(log_prob, dim=1)
                log_prob, word = log_prob.unsqueeze(1), word.unsqueeze(1)
            else:
                prob = torch.softmax((1 / temperature) * logits, dim=1)
                word = torch.multinomial(prob, 1)
                log_prob = log_prob.gather(1, word)

            # check for end-of-sentence tokens
            unfinished = word != self.special_symbols['eos']
            if i is 0:
                global_unfinished = unfinished
            else:
                global_unfinished = global_unfinished*unfinished

            # set loop variables
            previous_word = word
            word = word * global_unfinished.long()

            for lst, item in zip([question, log_probs, mask, hidden_arr, attention_arr],
                                 [word, log_prob, global_unfinished, hidden.squeeze(0).unsqueeze(1), attention.permute(0, 2, 1)]):
                lst.append(item)

        result = [torch.cat(x, dim=1) for x in [question, log_probs, mask, hidden_arr, attention_arr]]

        return Namespace(question=result[0], log_prob=result[1], mask=result[2], hidden=result[3], attention=result[4])

    def sample_beam(self, img_features, caption, pos, context, att_weights, q_idx_vec, beam_size, max_seq_len=17):
        batch_size = img_features.size(0)

        # encode the caption
        if self.use_position_encoding:
            input = torch.cat([self.caption_embedding(caption), q_idx_vec], dim=2)  # q_idx is position encoding vector
        else:
            input = self.caption_embedding(caption)

        _, cap_emb = self.caption_encoder(input)
        cap_emb = torch.cat([cap_emb[0], cap_emb[1]], dim=1)

        context_emb = torch.relu(self.context_embedding(context))
        pos_emb = torch.relu(torch.matmul(pos, self.pos_embedding.weight))

        img_features = self.vectorize_img(img_features)
        att_img_features = self.fc1(img_features)

        seq = torch.LongTensor(max_seq_len, batch_size).zero_()
        seqLogprobs = torch.FloatTensor(max_seq_len, batch_size)
        all_seq = torch.LongTensor(batch_size, beam_size, max_seq_len).zero_()
        all_seqLogprobs = torch.FloatTensor(batch_size, beam_size, max_seq_len).zero_()
        self.done_beams = [[] for _ in range(batch_size)]

        for i in range(batch_size):

            if self.use_caption_hidden:
                input = torch.cat([context_emb[i], pos_emb[i]], 0).unsqueeze(0).repeat(beam_size, 1)
            else:
                input = pos_emb[i].unsqueeze(0).repeat(beam_size, 1)

            hidden = self.fc2(input).unsqueeze(0)

            sample_img_feature = img_features[i].unsqueeze(0).repeat(beam_size, 1, 1)
            sample_att_feature = att_img_features[i].unsqueeze(0).repeat(beam_size, 1, 1)
            sample_cap_emb = cap_emb[i].unsqueeze(0).repeat(beam_size, 1)
            sample_att_weights = att_weights[i].unsqueeze(0).repeat(beam_size, 1)

            for t in range(1):
                if t == 0:  # input <bos>
                    previous_word = self.one_vec[:beam_size]  # beginning of sentence token is 1
                    q_emb = self.question_embedding(previous_word).squeeze()

                img_emb, _ = self.attention(sample_img_feature, sample_att_feature, hidden, sample_att_weights)

                logits, hidden = self.decoder(hidden, img_emb, sample_cap_emb, q_emb)
                state = (hidden, )  # state is the hidden, beam search expects a tuple

                log_probs = torch.log_softmax(logits, dim=1)

            self.done_beams[i] = self.beam_search(state, log_probs, sample_img_feature, sample_att_feature,
                                                  sample_cap_emb, sample_att_weights, beam_size, max_seq_len)
            seq[:, i] = self.done_beams[i][0]['seq']  # the first beam has highest cumulative score
            seqLogprobs[:, i] = self.done_beams[i][0]['logps']
            all_seq[i] = torch.stack([self.done_beams[i][k]['seq'] for k in range(beam_size)], dim=0)
            all_seqLogprobs[i] = torch.stack([self.done_beams[i][k]['logps'] for k in range(beam_size)], dim=0)
        # return the samples and their log likelihoods
        return seq.transpose(0, 1), seqLogprobs.transpose(0, 1), all_seq, all_seqLogprobs

    def get_logprobs_state(self, previous_word, img_features, att_img_features, sample_a_emb, state, att_weights):

        hidden = state[0]
        img_emb, _ = self.attention(img_features, att_img_features, hidden, att_weights)  # state is hidden
        q_emb = self.question_embedding(previous_word).squeeze(1)
        logits, hidden = self.decoder(hidden, img_emb, sample_a_emb, q_emb)
        state = (hidden, )

        return torch.log_softmax(logits, dim=1), state

    def beam_search(self, state, logprobs, sample_img_feature, sample_att_feature, sample_a_emb, att_weights, beam_size, max_seq_len):
        # args are the miscelleous inputs to the core in addition to embedded word and state
        # kwargs only accept opt

        def beam_step(logprobsf, beam_size, t, beam_seq, beam_seq_logprobs, beam_logprobs_sum, state):
            # INPUTS:
            # logprobsf: probabilities augmented after diversity
            # beam_size: obvious
            # t        : time instant
            # beam_seq : tensor contanining the beams
            # beam_seq_logprobs: tensor contanining the beam logprobs
            # beam_logprobs_sum: tensor contanining joint logprobs
            # OUPUTS:
            # beam_seq : tensor containing the word indices of the decoded captions
            # beam_seq_logprobs : log-probability of each decision made, same size as beam_seq
            # beam_logprobs_sum : joint log-probability of each beam

            ys, ix = torch.sort(logprobsf, 1, True)
            candidates = []
            cols = min(beam_size, ys.size(1))
            rows = beam_size
            if t == 0:
                rows = 1
            for c in range(cols):  # for each column (word, essentially)
                for q in range(rows):  # for each beam expansion
                    # compute logprob of expanding beam q with word in (sorted) position c
                    local_logprob = ys[q, c]
                    candidate_logprob = beam_logprobs_sum[q] + local_logprob
                    candidates.append({'c': ix[q, c], 'q': q, 'p': candidate_logprob, 'r': local_logprob})
            candidates = sorted(candidates, key=lambda x: -x['p'])

            new_state = [_.clone() for _ in state]
            # beam_seq_prev, beam_seq_logprobs_prev
            if t >= 1:
                # we''ll need these as reference when we fork beams around
                beam_seq_prev = beam_seq[:t].clone()
                beam_seq_logprobs_prev = beam_seq_logprobs[:t].clone()
            for vix in range(beam_size):
                v = candidates[vix]
                # fork beam index q into index vix
                if t >= 1:
                    beam_seq[:t, vix] = beam_seq_prev[:, v['q']]
                    beam_seq_logprobs[:t, vix] = beam_seq_logprobs_prev[:, v['q']]
                # rearrange recurrent states
                for state_ix in range(len(new_state)):
                    #  copy over state in previous beam q to new beam at vix
                    new_state[state_ix][:, vix] = state[state_ix][:, v['q']]  # dimension one is time step
                # append new end terminal at the end of this beam
                beam_seq[t, vix] = v['c']  # c'th word is the continuation
                beam_seq_logprobs[t, vix] = v['r']  # the raw logprob here
                beam_logprobs_sum[vix] = v['p']  # the new (sum) logprob along this beam
            state = new_state
            return beam_seq, beam_seq_logprobs, beam_logprobs_sum, state, candidates

        beam_seq = torch.LongTensor(max_seq_len, beam_size).zero_()
        beam_seq_logprobs = torch.FloatTensor(max_seq_len, beam_size).zero_()
        beam_logprobs_sum = torch.zeros(beam_size)  # running sum of logprobs for each beam
        done_beams = []

        for t in range(max_seq_len):
            """pem a beam merge. that is,
            for every previous beam we now many new possibilities to branch out
            we need to resort our beams to maintain the loop invariant of keeping
            the top beam_size most likely sequences."""
            logprobsf = logprobs.data.float()  # lets go to CPU for more efficiency in indexing operations
            # suppress UNK tokens in the decoding
            logprobsf[:, logprobsf.size(1) - 1] = logprobsf[:, logprobsf.size(1) - 1] - 1000

            beam_seq, \
            beam_seq_logprobs, \
            beam_logprobs_sum, \
            state, \
            candidates_divm = beam_step(logprobsf,
                                        beam_size,
                                        t,
                                        beam_seq,
                                        beam_seq_logprobs,
                                        beam_logprobs_sum,
                                        state)

            for vix in range(beam_size):
                # if time's up... or if end token is reached then copy beams
                if beam_seq[t, vix] == self.special_symbols['eos'] or t == max_seq_len - 1:
                    final_beam = {
                        'seq': beam_seq[:, vix].clone(),
                        'logps': beam_seq_logprobs[:, vix].clone(),
                        'p': beam_logprobs_sum[vix]
                    }
                    done_beams.append(final_beam)
                    # don't continue beams from finished sequences
                    beam_logprobs_sum[vix] = -1000

            # encode as vectors
            it = beam_seq[t]
            logprobs, state = self.get_logprobs_state(it.to(device), sample_img_feature, sample_att_feature, sample_a_emb, state, att_weights)

        done_beams = sorted(done_beams, key=lambda x: -x['p'])[:beam_size]
        return done_beams

    def vectorize_img(self, img_features):

        img_features = img_features.contiguous().view(-1, self.img_channels, self.img_area)  # Nx2048x49
        img_features = img_features.permute(0, 2, 1)
        return img_features