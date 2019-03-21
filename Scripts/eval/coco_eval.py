import os
import sys

import json
from json import encoder
encoder.FLOAT_REPR = lambda o: format(o, '.3f')

import numpy as np
import argparse
import pickle
from Utils.util import to_np

import torch
import torch.utils.data as data
from torch.utils.data import DataLoader
from Models.attention_captioner import AttentionCaptioner

sys.path.append("Dependencies/coco-caption")
from pycocotools.coco import COCO
from pycocoevalcap.eval import COCOEvalCap

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def parse_config():
    parser = argparse.ArgumentParser()
    parser.add_argument("--img_dir", type=str, help="path to where images are stored")
    parser.add_argument("--coco_annotation_file", type=str, help="path to coco-caption/annotations/captions_val2014.json")
    parser.add_argument("--eval_file", type=str, default="path to eval_val.p or eval_test.p")
    parser.add_argument("--model_file", type=str, help="checkpoint file for model to evaluate")
    parser.add_argument("--log_dir", type=str, help="where the predicted captions and scores should be saved")
    parser.add_argument("--split", type=str, default="val", help="val|test|train all 3 are handled")
    parser.add_argument("--batch_size", type=int, default=256, help='larger batch size means faster evaluation')
    parser.add_argument("--c_max_sentence_len", type=int, default=16, help="maximum sentence length")
    parser.add_argument("--eval_greedy", action='store_true', default=True, help='greedy decoding')
    parser.add_argument('--beam_size', type=int, default=3, help="beam size for evluation")

    return parser.parse_args()


class EvalDataset(data.Dataset):
    def __init__(self, opt, data_file, split="val"):

        self.f = pickle.load(open(data_file, "rb"))

        self.data = self.f["data"]
        self.img_dir = opt.img_dir
        self.split = split
        self.max_c_len = opt.c_max_sentence_len
        self.c_i2w, self.c_w2i = self.f["c_dicts"]
        self.c_vocab_size = len(self.c_i2w)
        self.special_symbols = self.f["special_symbols"]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        sample = self.data[index]

        img_id = sample['image_id']
        img_path = os.path.join(self.img_dir, "features", str(sample['image_id']) + '.npy')
        img = np.load(img_path)

        return img, img_id


def eval_coco(model, loader, run_name, log_dir, config):
    print("Computing coco-caption scores")

    # run coco-caption eval
    model.eval()

    with torch.no_grad():

        # compute predictions for each image
        predicted_captions = []

        for i, (image, image_id) in enumerate(loader):

            if config.beam_size < 2:
                # sample
                result = model.sample(image.to(device), greedy=config.eval_greedy, max_seq_len=loader.dataset.max_c_len+1)
                caps, cap_lens = to_np(result.caption), to_np(torch.sum(result.mask, dim=1))
            else:
                # beam search
                result = model.sample_beam(image.to(device), beam_size=config.beam_size, max_seq_len=loader.dataset.max_c_len + 1)
                # subtract 1 to exclude <eos> token which is returned by beam search
                caps, cap_lens = to_np(result.best_caption), to_np(torch.sum(result.best_logprobs != 0, dim=1)-1)

            # append captions to list
            for j in range(image.size(0)):

                caption = ' '.join([loader.dataset.c_i2w[x] for x in caps[j][:cap_lens[j]]]).strip()

                pred = {
                    'image_id': image_id[j].item(),
                    'caption': caption
                }

                predicted_captions.append(pred)

    # Reset
    model.train()
    return language_scores(predicted_captions, run_name, log_dir, annFile=config.coco_annotation_file)


def language_scores(preds, model_id, log_dir, annFile, split="val"):

    result_path = os.path.join(log_dir, 'cococap_scores_' + model_id + '_' + split + '.json')

    coco = COCO(annFile)
    valids = coco.getImgIds()

    # filter results to only those in MSCOCO validation set (will be about a third)
    preds_filt = [p for p in preds if p['image_id'] in valids]
    print('using %d/%d predictions' % (len(preds_filt), len(preds)))
    json.dump(preds_filt, open(result_path, 'w')) # serialize to temporary json file. Sigh, COCO API...

    cocoRes = coco.loadRes(result_path)
    cocoEval = COCOEvalCap(coco, cocoRes)
    cocoEval.params['image_id'] = cocoRes.getImgIds()
    cocoEval.evaluate()

    # create output dictionary
    out = {}
    for metric, score in cocoEval.eval.items():
        out[metric] = score

    imgToEval = cocoEval.imgToEval
    for p in preds_filt:
        image_id, caption = p['image_id'], p['caption']
        imgToEval[image_id]['caption'] = caption
    with open(result_path, 'w') as outfile:
        json.dump({'overall': out, 'imgToEval': imgToEval}, outfile)

    return out


if __name__ == "__main__":
    config = parse_config()

    dataset = EvalDataset(config, config.eval_file, config.split)
    loader = DataLoader(dataset, batch_size=config.batch_size, num_workers=1, shuffle=True)

    save_state = torch.load(config.model_file, map_location=lambda storage, loc: storage)

    model = AttentionCaptioner(save_state['opt']).to(device)
    model.load_state_dict(save_state['state_dict'])

    print eval_coco(model, loader, save_state['opt'].run_name, config.log_dir, config)
