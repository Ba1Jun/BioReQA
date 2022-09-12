export  CUDA_VISIBLE_DEVICES=1
python3 train_reqa.py \
    --seed 12345 \
    --do_train True \
    --do_test True \
    --dev_metric p1 \
    --dataset 6b \
    --max_question_len 24 \
    --max_answer_len 168 \
    --epoch 10 \
    --batch_size 32 \
    --model_type dual_encoder_wot \
    --encoder_type biobert \
    --plm_path /workspace/baijun/models/english/biobert-base-cased-v1.1 \
    --pooler_type mean \
    --matching_func cos \
    --whitening True \
    --temperature 0.05 \
    --learning_rate 5e-5 \
    --save_model_path output/6b/biobert_rbar/ \
    --load_model_path output/nli/biobert_ranking/model.pt \
    --rm_saved_model True \
    --save_results True \