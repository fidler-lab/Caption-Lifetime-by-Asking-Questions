import pdb
from Models.attention_captioner import AttentionCaptioner
from Models.attention_questioner_demo import QuestionGenerator
from Models.decision_maker import DecisionMaker
from Scripts.util import masked_softmax
from Utils.util import idx2str, init_state

import random
import numpy as np
import torch
import torchvision.models as models
import skimage.io
import pickle

from torchvision import transforms as trn
preprocess = trn.Compose([
        #trn.ToTensor(),
        trn.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

from Models.Resnet.resnet_utils import myResnet
import Models.Resnet.resnet as resnet

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

SEED = 600
CAPTION_PATH = "./Data/model_checkpoints/cap_start.pth"
QUESTION_PATH = "./Data/model_checkpoints/qgen_start.pth"
DM_PATH = "./Data/model_checkpoints/dm_start.pth"
RESNET_PATH = "./Utils/checkpoints/resnet101.pth"
VOCAB_DICT_PATH = "./Data/annotation/cap_64_train.p"

def load_model(path, model_class):

    save_state = torch.load(path, map_location=lambda storage, loc: storage)
    model = model_class(save_state['opt']).to(device)
    model.load_state_dict(save_state['state_dict'])

    print '{} model loaded at {}'.format(model_class, path)
    return model

captioner = load_model(CAPTION_PATH, AttentionCaptioner)
qgenerator = load_model(QUESTION_PATH, QuestionGenerator)
dmaker = load_model(DM_PATH, DecisionMaker)
d = pickle.load(open(VOCAB_DICT_PATH, "rb"))

ci2w, cw2i = d["c_dicts"][0], d["c_dicts"][1]
qi2w = d["q_dicts"][0]

net = getattr(resnet, "resnet101")()
net.load_state_dict(torch.load(RESNET_PATH))
encoder = myResnet(net)
encoder.to(device)
encoder.eval()

IMAGE_PATH = "./cat.jpg"

I = skimage.io.imread(IMAGE_PATH)
# handle grayscale input images
if len(I.shape) == 2:
    I = I[:,:,np.newaxis]
    I = np.concatenate((I,I,I), axis=2)

I = I.astype('float32')/255.0
I = torch.from_numpy(I.transpose([2,0,1]))
I = preprocess(I).to(device)
with torch.no_grad():
    features = encoder(I, 14)

  
image = features[1].permute(2, 1, 0).unsqueeze(0)    

# ========================================= Inference ======================================

def sample_decision(masked_prob, caption_mask, greedy=False):
    batch_size = masked_prob.size(0)
    zeros_mask = torch.zeros([batch_size, 17], dtype=torch.long, device=device, requires_grad=False)

    if greedy:
        masked_prob_copy = masked_prob.clone().detach()
        val, idx = torch.max(masked_prob_copy, 1)
        val, idx = val.unsqueeze(1), idx.unsqueeze(1)
    else:
        idx = torch.multinomial(masked_prob, 1)
        val = masked_prob.gather(1, idx)

    # edge-case: don't ask if probabilities are all 0s
    length = torch.clamp(torch.sum(caption_mask != 0, dim=1).long().unsqueeze(1), max=16)
    ask_flag = (val != 0) * (val > 0.0) * (idx != length)
    ask_mask = zeros_mask.scatter(1, idx, ask_flag.long()).detach()

    return [x.squeeze() for x in [val, idx, ask_flag, ask_mask]]

def pad_caption(caption, cap_len):
        # replace 0's beyond caption_length in captions with pad symbol int(c_vocab_size)
        range_matrix = torch.arange(17, dtype=torch.long, device=device).unsqueeze(0)
        padding = range_matrix >= cap_len.unsqueeze(1).repeat(1, 17)
        caption = caption + padding.long() * len(ci2w)
        return caption

# train or eval
captioner.train()
qgenerator.train()
dmaker.train()

# seed
set_seed(SEED)

# get original caption
r = captioner.sample(image, greedy=True, max_seq_len=17)

caption, cap_probs, cap_mask, pos_probs, att, topk_words, attended_img \
    = r.caption, r.prob, r.mask, r.pos_prob, r.attention.squeeze(), r.topk, r.atdimg

cap_len = cap_mask.long().sum(dim=1)
caption = pad_caption(caption, cap_len)

# get the hidden state context
ones_vector = torch.ones([1, 1], dtype=torch.long, device=device, requires_grad=False)
source = torch.cat([ones_vector, caption[:, :-1]], dim=1)
r = captioner(image, source, gt_pos=None, ss=False)
h = r.hidden

topk_words = [[y for y in x] for x in topk_words]

# 2. Identify the best time to ask a question, excluding ended sentences, baseline against the greedy decision
logit, valid_pos_mask = dmaker(h, attended_img, caption, cap_len,
                                pos_probs, topk_words, captioner.caption_embedding.weight.data)
masked_prob = masked_softmax(logit, cap_mask, valid_pos_mask, 1.0, max_len=16)

dm_prob, ask_idx, ask_flag, ask_mask = sample_decision(masked_prob, cap_mask, greedy=True)

# 3. Ask the teacher a question and get the answer
idx = ask_idx

pos_probs = pos_probs[0, idx]
h = h[0, idx]
att = att[idx]

# decision maker index vector
q_idx_vec = torch.zeros([1, 17, 256], dtype=torch.float, device=device, requires_grad=False)
q_idx_vec[0, idx, :] = 1.0

# query question generator
result = qgenerator.sample(image, caption, pos_probs.unsqueeze(0), h.unsqueeze(0), att.unsqueeze(0), q_idx_vec, greedy=True, max_seq_len=15, temperature=1.0)
question, q_logprob, q_mask = result.question, result.log_prob, result.mask
q_len = q_mask.long().sum(dim=1)


# get answer
answer = cw2i["squatting"]
answer_mask = torch.zeros([1, 17], dtype=torch.long, device=device, requires_grad=False)
answer_mask[0, idx] = answer

# get rollout caption
set_seed(SEED)
r = captioner.sample_with_teacher_answer(image, ask_mask.unsqueeze(0), answer_mask, torch.zeros([1, 1, 512], dtype=torch.float, device=device), torch.ones([1], dtype=torch.long, device=device), 17, True)
rollout, rollout_mask = r.caption, r.cap_mask
rollout_len = rollout_mask.long().sum(dim=1)

# get replace caption
replace = caption.clone()
replace[0, ask_idx] = answer

caption = caption[0, :cap_len].cpu().numpy()
rollout = rollout[0, :rollout_len].cpu().numpy()
replace = replace[0, :cap_len].cpu().numpy()
question = question[0, :q_len].cpu().numpy() 
ask_idx = ask_idx.item()

caption = ' '.join(idx2str(ci2w, caption))
rollout = ' '.join(idx2str(ci2w, rollout))
replace = ' '.join(idx2str(ci2w, replace))
question = ' '.join(idx2str(qi2w, question))

print(caption)
print(rollout)
print(replace)
print(question)
print(ask_idx)

