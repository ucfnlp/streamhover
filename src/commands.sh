# command

# train
python run_summarization.py --log_file ../logs/behance_train_c1024_e100.log --batch_size 20 --accum_count 20 
--mode train --label_smoothing 0 --num_min_word 5 --num_cluster 1024 --ch_dim 100 --epochs 30 
--sep_optim --warmup_steps_bert 3000 --warmup_steps_not_bert 1500 --save_checkpoint_steps 2000 --report_every 50

# test
python run_summarization.py --log_file ../logs/behance_test_c1024_e100.log --batch_size 100 --mode test --label_smoothing 0 
--train_from ../models/c1024_e100/model_c1024_e100.pt --num_min_word 5 --num_cluster 1024 --ch_dim 100
--num_sum_sent 3 4 5 -input_dist_level clip --is_sample_all --rouge_mean

# val_grid
python run_summarization.py --log_file ../logs/behance_val_grid_c1024_e100.log --batch_size 100 --mode val_grid --label_smoothing 0 
--train_from ../models/c1024_e100/model_c1024_e100.pt --num_min_word 5 --num_cluster 1024 --ch_dim 100 
--num_sum_sent 5 --input_dist_level clip --is_sample_all --rouge_mean

# test_grid
python run_summarization.py --log_file ../logs/behance_test_grid_1024_100.log --batch_size 100 --mode test_grid --label_smoothing 0 
--train_from ../models/c1024_e100/model_c1024_e100.pt --num_min_word 5 --num_cluster 1024 --ch_dim 100 
--num_sum_sent 5 --input_dist_level clip --is_sample_all --rouge_mean

# inference one clip
python run_summarization.py --log_file ../logs/behance_inf_clip_1024_100.log --batch_size 100 --mode inference_test --inference_cid 0
--label_smoothing 0 --train_from ../models/c1024_e100/model_c1024_e100.pt --num_min_word 5 --num_cluster 1024 --ch_dim 100 
--num_sum_sent 5 --is_sample_all

# inference video
python run_summarization.py --log_file ../logs/behance_inf_video_1024_100.log --mode video --video_inference_id 7 --video_inf_min_sent 30 --num_sum_sent 3
--batch_size 100 --label_smoothing 0 --train_from ../models/c1024_e100/model_c1024_e100.pt --num_min_word 5 --num_cluster 1024 --ch_dim 100
