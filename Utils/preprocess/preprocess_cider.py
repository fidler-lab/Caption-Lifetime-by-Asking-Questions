"""
Adapted from Ruotian Luo's implementation of self-critical policy gradient for image captioning
https://github.com/ruotianluo/self-critical.pytorch/blob/master

"""

import pickle
import argparse
from six.moves import cPickle
from collections import defaultdict
from tqdm import tqdm

def precook(s, n=4, out=False):
  """
  Takes a string as input and returns an object that can be given to
  either cook_refs or cook_test. This is optional: cook_refs and cook_test
  can take string arguments as well.
  :param s: string : sentence to be converted into ngrams
  :param n: int    : number of ngrams for which representation is calculated
  :return: term frequency vector for occuring ngrams
  """
  words = s.split()
  counts = defaultdict(int)
  for k in xrange(1,n+1):
    for i in xrange(len(words)-k+1):
      ngram = tuple(words[i:i+k])
      counts[ngram] += 1
  return counts

def cook_refs(refs, n=4): ## lhuang: oracle will call with "average"
    '''Takes a list of reference sentences for a single segment
    and returns an object that encapsulates everything that BLEU
    needs to know about them.
    :param refs: list of string : reference sentences for some image
    :param n: int : number of ngrams for which (ngram) representation is calculated
    :return: result (list of dict)
    '''
    return [precook(ref, n) for ref in refs]

def create_crefs(refs):
  crefs = []
  for ref in tqdm(refs):
    # ref is a list of 5 captions
    crefs.append(cook_refs(ref))
  return crefs

def compute_doc_freq(crefs):
  '''
  Compute term frequency for reference annotation.
  This will be used to compute idf (inverse document frequency later)
  The term frequency is stored in the object
  :return: None
  '''
  document_frequency = defaultdict(float)
  for refs in tqdm(crefs):
    # refs, k ref captions of one image
    for ngram in set([ngram for ref in refs for (ngram,count) in ref.iteritems()]):
      document_frequency[ngram] += 1
      # maxcounts[ngram] = max(maxcounts.get(ngram,0), count)
  return document_frequency

def build_dict(data, special_symbols, c_i2w):

    refs_idxs = []
    finished_img_ids = set()  # prevent redundancy

    for dat in tqdm(data):

        id = dat['image_id']

        if id in finished_img_ids:  # prevent 5x redundancy
            finished_img_ids.add(id)
            continue
        else:
            finished_img_ids.add(id)
            refs = []
            for caption in dat['captions']:
                idxs = caption['caption']
                words = ' '.join(c_i2w[x] for x in idxs)
                words += ' eos'
                words = words.strip()
                refs.append(words)
                # refs.append(' '.join(str(x) for x in idxs) + ' ' + str(special_symbols['eos']))

            refs_idxs.append(refs)

    ngram_idxs = compute_doc_freq(create_crefs(refs_idxs))
    return ngram_idxs, len(finished_img_ids)

def main(params):
  cap_file = params['data_file']
  f = pickle.load(open(cap_file, "rb"))
  data, _c_dicts, special_symbols = f["data"], f["c_dicts"], f["special_symbols"]

  ngram_idxs, ref_len = build_dict(data, special_symbols, _c_dicts[0])

  cPickle.dump({'document_frequency': ngram_idxs, 'ref_len': ref_len}, open(params['output_file']+'-idxs.p', 'w'), protocol=cPickle.HIGHEST_PROTOCOL)

if __name__ == "__main__":

  parser = argparse.ArgumentParser()

  # input json
  parser.add_argument('--data_file', default='Data/annotation/cap_train.p', help='input caption training file')
  parser.add_argument('--output_file', default='Data/annotation/coco-words', help='output ngram file')
  args = parser.parse_args()
  params = vars(args)  # convert to ordinary dict

  main(params)
