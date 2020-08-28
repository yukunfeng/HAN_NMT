set -x

extracted_dir="./zh-en-ted-extracted/"
preprocessed_dir="./zh-en-ted-preprocessed/"
converted_dir="./converted_zh_en_ted"

mkdir -p $converted_dir

python /home/lr/yukun/HAN_NMT/full_source/preprocess.py -train_src $preprocessed_dir/corpus.tc.zh -train_tgt $preprocessed_dir/corpus.tc.en -train_doc $extracted_dir/corpus.doc -valid_src $preprocessed_dir/IWSLT15.TED.dev2010.tc.zh -valid_tgt $preprocessed_dir/IWSLT15.TED.dev2010.tc.en -valid_doc $extracted_dir/IWSLT15.TED.dev2010.zh-en.doc -save_data $converted_dir -src_vocab_size 30000 -tgt_vocab_size 30000 -src_seq_length 80 -tgt_seq_length 80
