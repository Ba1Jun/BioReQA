export  CUDA_VISIBLE_DEVICES=1

python3 train_nli.py \
  --max_premise_len 64 \
  --max_hypothesis_len 32 \
  --epoch 3 \
  --batch_size 32 \
  --encoder_type biobert \
  --plm_path /workspace/baijun/models/english/biobert-base-cased-v1.1 \
  --pooler_type mean \
  --temperature 0.05 \
  --save_model output/nli/ \
  --learning_rate 2e-5 \
  --warmup_proportion 0.1 \
  --seed 42 \