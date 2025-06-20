python eval_title.py \
    --model_path "distilled model path" \
    --data_path "TIM_Eval_title.json"\
    --output_path "./results/title/" \
    --batch_size 16 \
    --device 0 
	
python eval_project.py \
    --model_path "distilled model path" \
    --data_path ".TIM_Eval_project.json"\
    --output_path "./results/project/" \
    --batch_size 16 \
    --device 1