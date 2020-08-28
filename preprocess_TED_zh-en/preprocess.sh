#!/bin/bash
# moses_scripts=/idiap/user/lmiculicich/Installations/mosesdecoder/scripts
moses_scripts=/home/lr/yukun/OpenNMT-py/data/mosesdecoder/scripts/

# zh_segment_home=/idiap/home/lmiculicich/.cache/pip/wheels/ce/32/de/c2be1db5f30804bc7f146ff698c52963f8aa11ba5049811b0d
#kpu_preproc_dir=/fs/zisa0/bhaddow/code/preprocess/build/bin

max_len=200

export PYTHONPATH=$zh_segment_home

src=zh
tgt=en
pair=$src-$tgt
input_dir="./zh-en-ted-extracted/"
output_dir="./zh-en-ted-preprocessed/"

mkdir -p $output_dir

# Tokenise the English part
cat $input_dir/corpus.$tgt | \
$moses_scripts/tokenizer/normalize-punctuation.perl -l $tgt | \
$moses_scripts/tokenizer/tokenizer.perl -a -l $tgt  \
> $output_dir/corpus.tok.$tgt

#Segment the Chinese part
python -m jieba -d ' ' < $input_dir/corpus.$src > $output_dir/corpus.tc.$src 

#
###
#### Clean
#$moses_scripts/training/clean-corpus-n.perl corpus.tok $src $tgt corpus.clean 1 $max_len corpus.retained
###
#

#### Train truecaser and truecase
$moses_scripts/recaser/train-truecaser.perl -model $output_dir/truecase-model.$tgt -corpus $output_dir/corpus.tok.$tgt
$moses_scripts/recaser/truecase.perl < $output_dir/corpus.tok.$tgt > $output_dir/corpus.tc.$tgt -model $output_dir/truecase-model.$tgt

# rm -f $output_dir/corpus.tc.$src
# ln -s $output_dir/corpus.tok.$src  $output_dir/corpus.tc.$src
#
#  
# dev sets
for devset in dev2010 tst2010 tst2011 tst2012 tst2013; do
  for lang  in $src $tgt; do
    if [ $lang = $tgt ]; then
      side="src"
      $moses_scripts/tokenizer/normalize-punctuation.perl -l $lang < $input_dir/IWSLT15.TED.$devset.$src-$tgt.$lang | \
      $moses_scripts/tokenizer/tokenizer.perl -a -l $lang |  \
      $moses_scripts/recaser/truecase.perl -model $output_dir/truecase-model.$lang \
      > $output_dir/IWSLT15.TED.$devset.tc.$lang
    else
      side="ref"
      python -m jieba -d ' '  < $input_dir/IWSLT15.TED.$devset.$src-$tgt.$lang > $output_dir/IWSLT15.TED.$devset.tc.$lang
    fi
    
  done

done
