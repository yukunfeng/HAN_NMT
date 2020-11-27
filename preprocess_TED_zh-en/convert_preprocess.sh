set -x

extracted_dir="./zh-en-ted-extracted/"
preprocessed_dir="./zh-en-ted-preprocessed/"
converted_dir="./converted_zh_en_ted"
mkdir -p $converted_dir
output_base="${converted_dir}/zh_en_ted"

python /home/lr/yukun/HAN_NMT/full_source/preprocess.py -train_src $preprocessed_dir/corpus.tc.zh -train_tgt $preprocessed_dir/corpus.tc.en -train_doc $extracted_dir/corpus.doc -valid_src $preprocessed_dir/IWSLT15.TED.dev2010.tc.zh -valid_tgt $preprocessed_dir/IWSLT15.TED.dev2010.tc.en -valid_doc $extracted_dir/IWSLT15.TED.dev2010.zh-en.doc -save_data $output_base -src_vocab_size 30000 -tgt_vocab_size 30000 -src_seq_length 80 -tgt_seq_length 80

# train
# ./zh-en-ted-preprocessed/corpus.tc.zh
# ./zh-en-ted-preprocessed/corpus.tc.en

# valid
# ./zh-en-ted-preprocessed/IWSLT15.TED.dev2010.tc.zh
# ./zh-en-ted-preprocessed/IWSLT15.TED.dev2010.tc.en

# test
# ./zh-en-ted-preprocessed/IWSLT15.TED.tst2010_2013.tc.zh
# ./zh-en-ted-preprocessed/IWSLT15.TED.tst2010_2013.tc.en

