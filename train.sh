set -x

data="./preprocess_TED_zh-en/converted_zh_en_ted/zh_en_ted"
model="./saved_models/"
mkdir -p $model
model="${model}/$(basename $data).sent"

python ./full_source/train.py -data $data \
    -save_model $model \
    -encoder_type transformer -decoder_type transformer \
    -enc_layers 6 -dec_layers 6 -label_smoothing 0.1 \
    -src_word_vec_size 512 -tgt_word_vec_size 512 \
    -rnn_size 512 -position_encoding -dropout 0.1 \
    -batch_size 4096 -start_decay_at 20 -report_every 500 \
    -epochs 20 -gpuid 0 -max_generator_batches 16 \
    -batch_type tokens -normalization tokens -accum_count 4 \
    -optim adam -adam_beta2 0.998 -decay_method noam \
    -warmup_steps 8000 -learning_rate 2 -max_grad_norm 0 \
    -param_init 0 -param_init_glorot \
    -train_part sentences
