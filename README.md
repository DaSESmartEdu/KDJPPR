# KDJPPR
Knowledge Distillation for Job Title Prediction and Project Recommendation in Open Source Communities

## **Environment Setup**

The experiments are conducted in parallel on **2 NVIDIA A800 80G GPUs**.

### **Installation**

Run the following commands to set up the environment:

```bash
conda create -n your_project_name python=3.10
conda activate your_project_name
pip install -r requirements.txt
```

## **Project Structure**

```
KDJPPR/
│── src/
│   ├── data_process/
│   │  ├── step_1_extract.py
│   │  ├── step_2_get_embeddings.py
│   │  ├── step_3_get_candidates.py
│   │  ├── step_4_prompt_API_skill.py
│   │  ├── step_4_prompt_title.py
│   │  ├── step_5_merge.py
│   │  ├── script.sh
│   ├── distillation/
│   │  ├── distill_train.py
│   │  ├── ds_config_zero2.json
│   │  ├── script.sh
│   ├── evaluate/
│   │  ├── eval_project.py
│   │  ├── eval_title.py
│   │  ├── merge.py
│   │  ├── TM-Eval_project_data.json
│   │  ├── TM-Eval_title_data.json
│   ├── finetune/
│   │  ├── merge.py
│   │  ├── ds_config_zero2.json
│   │  ├── finetune.py
│   │  ├── script.sh
│   ├── LLM_download.py
│── README.md
```

## **Usage**  

### **Download LLMs**

To download the required LLMs, run:

```bash
python LLM_download.py --model_name <model_name> --cache_dir <model_save_path>
```

### **Data Processing**

All data processing scripts are located in the `src/data_process/` directory. The JA-QA dataset is available on [here](https://drive.usercontent.google.com/download?id=1FgztW_EqfwfAKvXH87GBhZxQwqlx5W7c&export=download&authuser=0)

### **Fine-tune the Model**  

To fine-tune the teacher model, refer to `src/finetune/script.sh`. After fine-tuning, run `src/finetune/merge.py` to merge the fine-tuned model.

### **Distill the Model**  

Using the fine-tuned teacher LLMs, refer to `src/distillation/script.sh` to distill the student model. After distillation, run `src/distillation/merge.py` to merge the distilled student model.

### **Evaluate the Model**  

To evaluate the performance of the distilled student model, refer to `src/evaluate/script.sh`.
