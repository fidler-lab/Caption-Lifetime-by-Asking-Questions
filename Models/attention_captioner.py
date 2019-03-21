import torch
import torch.nn as nn
from Utils.util import Bunch

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
        self.fc3 = nn.Linear(self.img_channels, self.img_channels)

    def forward(self, img_features, att_img_features, hidden):

        z = self.fc1(hidden)
        z = z.permute(1, 0, 2).repeat(1, self.img_area, 1)  # Nx49x512
        z = att_img_features + z
        z = torch.tanh(z)
        z = self.fc2(z)
        a = torch.softmax(z, dim=1)

        attended_feature = (img_features * a).sum(1)
        attended_feature = torch.relu(self.fc3(attended_feature))

        return attended_feature, a


class Decoder(nn.Module):

    def __init__(self, opt):

        super(Decoder, self).__init__()

        # vocab sizes
        self.c_vocab_size = opt.c_vocab_size
        self.pos_vocab_size = opt.pos_vocab_size

        # embedding sizes
        self.emb_size = opt.word_embedding_size
        self.pos_emb_size = opt.pos_emb_size

        # architecture sizes
        self.rnn_hidden_size = opt.rnn_size
        self.rnn_num_layers = opt.rnn_layers

        # image dimensions
        self.img_channels = opt.image_channels

        # dropout and scheduled sampling
        self.dropout_rate = opt.dropout
        self.pos_ss_prob = opt.p_scheduled_sampling_initial_value

        # pos embedding
        self.pos_embedding = nn.Embedding(self.pos_vocab_size+1, self.pos_emb_size, padding_idx=self.pos_vocab_size)

        self.gru = nn.GRU(self.emb_size+self.img_channels, self.rnn_hidden_size, self.rnn_num_layers,
                          batch_first=True, dropout=self.dropout_rate)

        self.fc1 = nn.Linear(self.rnn_hidden_size, self.pos_vocab_size+1)
        self.fc2 = nn.Linear(self.rnn_hidden_size+self.pos_emb_size, self.c_vocab_size+1)
        self.dropout = nn.Dropout(self.dropout_rate)

    def forward(self, hidden, img_emb, c_emb, gt_pos):

        hidden, last_hidden = self.gru(torch.cat((img_emb, c_emb), 1).unsqueeze(1), hidden)

        z = self.dropout(hidden.squeeze(1))
        pos_logits = self.fc1(z)

        # scheduled sampling for POS
        sample_prob = torch.Tensor(1).uniform_(0, 1)

        if gt_pos is None or sample_prob[0] < self.pos_ss_prob:
            pos_emb = torch.matmul(torch.softmax(pos_logits, dim=1), self.pos_embedding.weight).detach()
        else:
            pos_emb = self.pos_embedding(gt_pos).squeeze(1)

        logits = self.fc2(torch.cat((z, pos_emb), 1))

        return logits, last_hidden, pos_logits


def init_hidden(batch_size, rnn_size, num_layers=1):
    return torch.zeros([num_layers, batch_size, rnn_size], dtype=torch.float32, device=device)


class AttentionCaptioner(nn.Module):

    def __init__(self, opt):

        super(AttentionCaptioner, self).__init__()

        # vocab sizes
        self.c_vocab_size = opt.c_vocab_size
        self.pos_vocab_size = opt.pos_vocab_size
        self.special_symbols = opt.special_symbols

        # architecture sizes
        self.att_hidden_size = opt.att_hidden_size
        self.rnn_hidden_size = opt.rnn_size
        self.rnn_num_layers = opt.rnn_layers

        # embeding sizes
        self.emb_size = opt.word_embedding_size

        # image dimensions
        self.img_channels = opt.image_channels
        self.img_width = opt.image_feature_size
        self.img_area = self.img_width * self.img_width

        # top k for MK module and schedule sampling
        self.k = opt.k
        self.ss_prob = 0.0

        # dummy vector
        self.one_vec = torch.ones([5*opt.batch_size, 1], dtype=torch.long, device=device)

        # submodules
        self.attention = Attention(opt)
        self.decoder = Decoder(opt)

        # embedding layers
        self.caption_embedding = nn.Embedding(self.c_vocab_size+1, self.emb_size, padding_idx=self.c_vocab_size)

        # FC layers
        self.fc1 = nn.Linear(self.img_channels, self.att_hidden_size)

        # init word embeddings
        initrange = 0.1
        self.caption_embedding.weight.data[:-1].uniform_(-initrange, initrange)  # -1 to deal with pad index

    def vectorize_img(self, img_features):

        img_features = img_features.contiguous().view(-1, self.img_channels, self.img_area)  # Nx2048x49
        img_features = img_features.permute(0, 2, 1)
        return img_features

    def forward(self, img_features, source, gt_pos, ss=True):
        batch_size, seq_length = source.size()[:2]

        img_features = self.vectorize_img(img_features)
        img_att_features = self.fc1(img_features)

        hidden = init_hidden(batch_size, self.rnn_hidden_size)
        logits_arr, pos_logits_arr, hidden_arr, att_arr = [], [], [], []

        for i in range(seq_length):

            # scheduled sampling
            if ss and i >= 1 and self.ss_prob > 0.0:
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
            img_emb, att_weights = self.attention(img_features, img_att_features, hidden)
            c_emb = self.caption_embedding(previous_word)

            # decode caption logits
            if gt_pos is None:
                logits, hidden, pos_logits = self.decoder(hidden, img_emb, c_emb, None)
            else:
                logits, hidden, pos_logits = self.decoder(hidden, img_emb, c_emb, gt_pos[:, i])

            # collect variables we will be returning
            logits_arr.append(logits.unsqueeze(1))
            pos_logits_arr.append(pos_logits.unsqueeze(1))
            hidden_arr.append(hidden.permute(1, 0, 2))
            att_arr.append(att_weights.permute(0, 2, 1))

        result = [torch.cat(x, dim=1) for x in [logits_arr, pos_logits_arr, hidden_arr, att_arr]]

        return Bunch(logits=result[0], pos_logits=result[1], hidden=result[2], att=result[3])

    def sample_with_teacher_answer(self, image, ask_mask, answer_mask, hidden, previous_word, max_len, greedy, temperature=1.0):

        h_arr, pos_arr, caption, cap_mask, capprob_arr, att_arr = [], [], [], [], [], []

        # calculate image features
        img_features = self.vectorize_img(image)
        att_img_features = self.fc1(img_features)

        for t in range(max_len):
            # attend on image
            img_emb, att = self.attention(img_features, att_img_features, hidden)
            # embed previous word
            c_emb = self.caption_embedding(previous_word)
            # rollout rnn decoder
            logits, hidden, pos_logits = self.decoder(hidden, img_emb, c_emb, None)
            word_probs = torch.softmax((1 / temperature) * logits, dim=1)
            log_prob = torch.log_softmax((1 / temperature) * logits, dim=1)
            pos_probs = torch.softmax(pos_logits, dim=1)

            # sample the next word
            if greedy:
                chosen_log_prob, word = torch.max(log_prob, dim=1)
                chosen_log_prob, word = chosen_log_prob.unsqueeze(1), word.unsqueeze(1)
            else:
                word = torch.multinomial(word_probs, 1)
                chosen_log_prob = log_prob.gather(1, word)

            # set previous word either to answer or word in caption
            previous_word = word.squeeze(1) * (ask_mask[:, t] == 0).long() + answer_mask[:, t] * ask_mask[:, t]

            unfinished = previous_word != self.special_symbols['eos']
            global_unfinished = unfinished if t is 0 else global_unfinished * unfinished

            [arr.append(x.unsqueeze(1)) for arr, x in zip(
                [h_arr, pos_arr, caption, cap_mask, att_arr, capprob_arr],
                 [hidden.squeeze(), pos_probs, previous_word, global_unfinished, att, word_probs])]

        result = [torch.cat(x, dim=1) for x in [h_arr, pos_arr, caption, cap_mask, att_arr, capprob_arr]]

        return Bunch(hidden=result[0], pos_prob=result[1], caption=result[2], cap_mask=result[3], att=result[4],
                     cap_prob=result[5])

    # def one_time_step(self, img_features, hidden, previous_word, temperature=1.0, greedy=False):
    #
    #     # calculate image features
    #     img_features = self.vectorize_img(img_features)
    #     att_img_features = self.fc1(img_features)
    #
    #     # attend on image
    #     img_emb, att = self.attention(img_features, att_img_features, hidden)
    #     # embed previous word
    #     c_emb = self.caption_embedding(previous_word)
    #
    #     # rollout rnn decoder
    #     logits, hidden, pos_logits = self.decoder(hidden, img_emb, c_emb, gt_pos)
    #
    #     pos_probs = self.softmax1(pos_logits)
    #     word_probs = self.softmax1((1 / temperature) * logits)
    #     log_prob = F.log_softmax((1 / temperature) * logits, dim=1)
    #     pos_log_prob = F.log_softmax(pos_logits, dim=1)
    #
    #     # sample the next word
    #     if greedy:
    #         chosen_log_prob, word = torch.max(log_prob, dim=1)
    #         chosen_log_prob, word = chosen_log_prob.unsqueeze(1), word.unsqueeze(1)
    #     else:
    #         word = torch.multinomial(word_probs, 1)
    #         chosen_log_prob = log_prob.gather(1, word)
    #
    #     topk = torch.topk(word_probs, k=self.sample_topk, dim=1)
    #     word_entropy = torch.sum(-word_probs * log_prob, dim=1)
    #     pos_entropy = torch.sum(-pos_probs * pos_log_prob, dim=1)
    #
    #     return Bunch(h=hidden, pos_probs=pos_probs, word=word.squeeze(1), word_probs=word_probs, log_prob=chosen_log_prob.squeeze(1), att=att.squeeze(2), word_entropy=word_entropy, pos_entropy=pos_entropy, topk=topk)

    def sample(self, img_features, greedy=False, max_seq_len=17, temperature=1.0, hidden=None, previous_word=None):
        batch_size = img_features.size(0)

        # initialize hidden state and previous input
        if hidden is None:
            hidden = init_hidden(batch_size, self.rnn_hidden_size)
        if previous_word is None:
            previous_word = self.one_vec[:batch_size]  # beginning of sentence token

        img_features = self.vectorize_img(img_features)
        img_att_features = self.fc1(img_features)
        caption, prob_arr, lp_arr, mask, hidden_arr, posprob_arr, attention_arr, topk, atdimg_arr = [], [], [], [], [], [], [], [], []

        for i in range(max_seq_len):
            # compute attended image feature and word embeddings
            img_emb, attention = self.attention(img_features, img_att_features, hidden)
            c_emb = self.caption_embedding(previous_word).squeeze(1)

            # decode caption logits
            logits, hidden, pos_logits = self.decoder(hidden, img_emb, c_emb, None)

            # sample a word
            prob = torch.softmax((1 / temperature) * logits, dim=1)
            log_prob = torch.log_softmax((1 / temperature) * logits, dim=1)

            if greedy:
                chosen_log_prob, word = torch.max(log_prob, dim=1)
                chosen_log_prob, word = chosen_log_prob.unsqueeze(1), word.unsqueeze(1)
            else:
                word = torch.multinomial(prob, 1)
                chosen_log_prob = log_prob.gather(1, word)

            # pad sentences that have ended
            unfinished = word != self.special_symbols['eos']
            if i is 0:
                global_unfinished = unfinished
            else:
                global_unfinished = global_unfinished*unfinished

            previous_word = word
            word = word * global_unfinished.long()

            # collect variables we will be returning
            caption.append(word)
            lp_arr.append(chosen_log_prob)
            mask.append(global_unfinished)
            topk.append(torch.topk(prob, k=self.k, dim=1))
            [arr.append(x.unsqueeze(1)) for arr,x in zip([prob_arr, hidden_arr, posprob_arr, attention_arr, atdimg_arr],
                                                         [prob, hidden.squeeze(0), torch.softmax(pos_logits, dim=1), attention, img_emb])]

        result = [torch.cat(x, dim=1) for x in [caption, prob_arr, lp_arr, mask, hidden_arr, posprob_arr, attention_arr, atdimg_arr]]

        return Bunch(caption=result[0], prob=result[1], log_prob=result[2], mask=result[3], hidden=result[4], pos_prob=result[5], attention=result[6], atdimg=result[7], topk=topk)

    def sample_beam(self, img_features, beam_size, max_seq_len=17):
        batch_size = img_features.size(0)

        img_features = self.vectorize_img(img_features)
        img_featuresx = self.fc1(img_features)

        best_caption = torch.LongTensor(max_seq_len, batch_size).zero_()
        best_logprob = torch.FloatTensor(max_seq_len, batch_size)
        captions = torch.LongTensor(batch_size, beam_size, max_seq_len).zero_()
        logprobs = torch.FloatTensor(batch_size, beam_size, max_seq_len).zero_()
        self.done_beams = [[] for _ in range(batch_size)]

        for i in range(batch_size):

            hidden = init_hidden(beam_size, self.rnn_hidden_size)

            sample_img_feature = img_features[i].unsqueeze(0).repeat(beam_size, 1, 1)
            sample_att_feature = img_featuresx[i].unsqueeze(0).repeat(beam_size, 1, 1)

            for t in range(1):
                if t == 0:  # input <bos>
                    assert self.special_symbols['bos'] == 1
                    previous_word = self.one_vec[:beam_size]  # beginning of sentence token is 1
                    c_emb = self.caption_embedding(previous_word).squeeze(1)

                img_emb, _ = self.attention(sample_img_feature, sample_att_feature, hidden)

                logits, hidden, _ = self.decoder(hidden, img_emb, c_emb, None)
                state = (hidden, )  # state is the hidden, beam search expects a tuple

                log_probs = torch.log_softmax(logits, dim=1)

            self.done_beams[i] = self.beam_search(state, log_probs, sample_img_feature, sample_att_feature, beam_size, max_seq_len)
            best_caption[:, i] = self.done_beams[i][0]['seq']  # the first beam has highest cumulative score
            best_logprob[:, i] = self.done_beams[i][0]['logps']
            captions[i] = torch.stack([self.done_beams[i][k]['seq'] for k in range(beam_size)], dim=0)
            logprobs[i] = torch.stack([self.done_beams[i][k]['logps'] for k in range(beam_size)], dim=0)
        # return the samples and their log likelihoods
        return Bunch(best_caption=best_caption.transpose(0, 1), best_logprob=best_logprob.transpose(0, 1), captions=captions, logprobs=logprobs)

    def get_logprobs_state(self, previous_word, img_features, att_img_features, state):

        hidden = state[0]
        img_emb, _ = self.attention(img_features, att_img_features, hidden)  # state is hidden
        c_emb = self.caption_embedding(previous_word).squeeze(1)
        logits, hidden, _ = self.decoder(hidden, img_emb, c_emb, None)
        state = (hidden, )

        return torch.log_softmax(logits, dim=1), state

    def beam_search(self, state, logprobs, sample_img_feature, sample_att_feature, beam_size, max_seq_len):
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
            logprobs, state = self.get_logprobs_state(it.to(device), sample_img_feature, sample_att_feature, state)

        done_beams = sorted(done_beams, key=lambda x: -x['p'])[:beam_size]
        return done_beams
