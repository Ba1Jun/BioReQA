export  CUDA_VISIBLE_DEVICES=0
python3 train_nli_ranking.py \
  --max_premise_len 64 \
  --max_hypothesis_len 32 \
  --epoch 3 \
  --batch_size 32 \
  --model_type dual_encoder_wot \
  --encoder_type biobert \
  --plm_path /workspace/baijun/models/english/biobert-base-cased-v1.1 \
  --matching_func cos \
  --pooler_type mean \
  --temperature 0.05 \
  --save_model_path output/nli/biobert_ranking \
  --learning_rate 2e-5 \
  --warmup_proportion 0.1 \
  --seed 12345