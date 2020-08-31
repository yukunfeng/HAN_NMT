set -x

data="./preprocess_TED_zh-en/converted_zh_en_ted/zh_en_ted"
model_dir="./saved_models"
mkdir -p $model_dir
model="${model_dir}/$(basename $data).sent"

# python ./full_source/train.py -data $data \
    # -save_model $model \
    # -encoder_type transformer -decoder_type transformer \
    # -enc_layers 6 -dec_layers 6 -label_smoothing 0.1 \
    # -src_word_vec_size 512 -tgt_word_vec_size 512 \
    # -rnn_size 512 -position_encoding -dropout 0.1 \
    # -batch_size 4096 -start_decay_at 20 -report_every 500 \
    # -epochs 26 -gpuid 0 -max_generator_batches 16 \
    # -batch_type tokens -normalization tokens -accum_count 4 \
    # -optim adam -adam_beta2 0.998 -decay_method noam \
    # -warmup_steps 8000 -learning_rate 2 -max_grad_norm 0 \
    # -param_init 0 -param_init_glorot \
    # -train_part sentences \
    # -start_checkpoint_at 20


# Train enc increasingly
han_enc_model="$model_dir/$(basename $data).han.enc"
sent_exact_model=$(ls ${model}*e19.pt)
# python ./full_source/train.py -data $data \
    # -save_model $han_enc_model \
    # -encoder_type transformer -decoder_type transformer \
    # -enc_layers 6 -dec_layers 6 -label_smoothing 0.1 \
    # -src_word_vec_size 512 -tgt_word_vec_size 512 -rnn_size 512 \
    # -position_encoding -dropout 0.1 -batch_size 1024 \
    # -start_decay_at 2 -report_every 500 -epochs 1 -gpuid 0 \
    # -max_generator_batches 32 -batch_type tokens \
    # -normalization tokens -accum_count 4 \
    # -optim adam -adam_beta2 0.998 -decay_method noam -warmup_steps 8000 \
    # -learning_rate 2 -max_grad_norm 0 -param_init 0 -param_init_glorot \
    # -train_part all -context_type HAN_enc -context_size 3 \
    # -train_from $sent_exact_model \
    # -start_checkpoint_at 1


# Train joint increasingly
han_joint_model="$model_dir/$(basename $data).joint"
han_enc_exact_name=$(ls ${han_enc_model}*e1.pt)
# python ./full_source/train.py -data $data \
    # -save_model $han_joint_model \
    # -encoder_type transformer -decoder_type transformer \
    # -enc_layers 6 -dec_layers 6 -label_smoothing 0.1 -src_word_vec_size 512 \
    # -tgt_word_vec_size 512 -rnn_size 512 -position_encoding \
    # -dropout 0.1 -batch_size 1024 -start_decay_at 2 \
    # -report_every 500 -epochs 1 -gpuid 0 -max_generator_batches 32 \
    # -batch_type tokens -normalization tokens -accum_count 4 \
    # -optim adam -adam_beta2 0.998 -decay_method noam -warmup_steps 8000 \
    # -learning_rate 2 -max_grad_norm 0 -param_init 0 -param_init_glorot \
    # -train_part all -context_type HAN_join -context_size 3 \
    # -train_from $han_enc_exact_name \
    # -start_checkpoint_at 1


# Translate
function translate() {
    local part=$1
    local translate_model=$2 
    local note=$3

    translate_src="./preprocess_TED_zh-en/zh-en-ted-preprocessed/IWSLT15.TED.tst2010_2013.tc.zh"
    translate_tgt="./preprocess_TED_zh-en/zh-en-ted-preprocessed/IWSLT15.TED.tst2010_2013.tc.en"
    translate_doc="./preprocess_TED_zh-en/zh-en-ted-extracted/IWSLT15.TED.tst2010_2013.zh-en.doc"
    translate_out="$(basename $translate_tgt).by.$(basename $translate_model).txt"
    bleu_script=/home/lr/yukun/OpenNMT-py/data/mosesdecoder/scripts/generic/multi-bleu.perl

    if [ "$part" = "sent"  ]; then 
        python ./full_source/translate.py -model $translate_model \
            -src $translate_src -tgt $translate_tgt -doc $translate_doc \
            -output $translate_out -translate_part sentences -batch_size 32 -gpu 0
    else
        python ./full_source/translate.py -model $translate_model \
            -src $translate_src -tgt $translate_tgt -doc $translate_doc \
            -output $translate_out -translate_part all -batch_size 32 -gpu 0
    fi;

    perl $bleu_script $translate_tgt < $translate_out
    epoch "above result is from $(basename $translate_model)"
}

python ~/env_config/sending_emails.py -c  "$0 succ: $? NMT finished;"
