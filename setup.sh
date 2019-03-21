#!/bin/bash
cd ./Dependencies

export PYTHONPATH=$PWD:$PWD/Dependencies/coco-caption:$PWD/Dependencies/cider

pip install -r requirements.txt
if [ ! -d stanford-corenlp-full-2017-06-09 ]; then
  wget -N http://nlp.stanford.edu/software/stanford-corenlp-full-2017-06-09.zip
  unzip stanford-corenlp-full-2017-06-09.zip;
  rm stanford-corenlp-full-2017-06-09.zip;
fi

git clone https://github.com/shenkev/coco-caption.git
git clone https://github.com/shenkev/cider.git