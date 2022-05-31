export  CUDA_VISIBLE_DEVICES=1
python3 train_reqa.py \
      --do_test \
      --rm_saved_model \
      --dataset 9b \
      --max_question_len 32 \
      --max_answer_len 256 \
      --epoch 10 \
      --batch_size 32 \
      --model_type dual_encoder \
      --encoder_type biobert \
      --plm_path /workspace/baijun/models/english/biobert-base-cased-v1.1 \
      --pooler_type mean \
      --temperature 0.05 \
      --learning_rate 2e-5 \
      --save_model output/baseline \