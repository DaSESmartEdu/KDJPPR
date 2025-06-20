from dataclasses import dataclass, field
import json
import math
import logging
import os
from typing import Dict, Optional, List
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from deepspeed import zero
from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus
import transformers
from transformers import Trainer, GPTQConfig
from transformers.integrations import deepspeed
from transformers.trainer_pt_utils import LabelSmoother
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from accelerate.utils import DistributedType
from tqdm import tqdm
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
IGNORE_TOKEN_ID = LabelSmoother.ignore_index


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="Qwen/Qwen-7B")
    teacher_model_path: Optional[str] = field(default=None)

@dataclass
class DistillationArguments:
    temperature: float = field(default=3.0)
    alpha: float = field(default=0.5)  
    beta: float = field(default=0.5)  

@dataclass
class DataArguments:
    data_path: str = field(
        default=None, metadata={"help": "Path to the training data."}
    )
    eval_data_path: str = field(
        default=None, metadata={"help": "Path to the evaluation data."}
    )
    lazy_preprocess: bool = False


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(
        default=8192,
        metadata={
            "help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )
    use_lora: bool = False


@dataclass
class LoraArguments:
    lora_r: int = 64
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_target_modules: List[str] = field(
        default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    )
    lora_weight_path: str = ""
    lora_bias: str = "none"
    q_lora: bool = False


def maybe_zero_3(param):
    if hasattr(param, "ds_id"):
        assert param.ds_status == ZeroParamStatus.NOT_AVAILABLE
        with zero.GatheredParameters([param]):
            param = param.data.detach().cpu().clone()
    else:
        param = param.detach().cpu().clone()
    return param



def get_peft_state_maybe_zero_3(named_params, bias):
    if bias == "none":
        to_return = {k: t for k, t in named_params if "lora_" in k}
    elif bias == "all":
        to_return = {k: t for k, t in named_params if "lora_" in k or "bias" in k}
    elif bias == "lora_only":
        to_return = {}
        maybe_lora_bias = {}
        lora_bias_names = set()
        for k, t in named_params:
            if "lora_" in k:
                to_return[k] = t
                bias_name = k.split("lora_")[0] + "bias"
                lora_bias_names.add(bias_name)
            elif "bias" in k:
                maybe_lora_bias[k] = t
        for k, t in maybe_lora_bias:
            if bias_name in lora_bias_names:
                to_return[bias_name] = t
    else:
        raise NotImplementedError
    to_return = {k: maybe_zero_3(v) for k, v in to_return.items()}
    return to_return


local_rank = None

def rank0_print(*args):
    if local_rank == 0:
        print(*args)


def safe_save_model_for_hf_trainer(trainer: transformers.Trainer, output_dir: str, bias="none"):

    if deepspeed.is_deepspeed_zero3_enabled():
        state_dict = trainer.model_wrapped._zero3_consolidated_16bit_state_dict()
    else:
        if trainer.args.use_lora:
            state_dict = get_peft_state_maybe_zero_3(
                trainer.model.named_parameters(), bias
            )
        else:
            state_dict = trainer.model.state_dict()
    if trainer.args.should_save and trainer.args.local_rank == 0:
        trainer._save(output_dir, state_dict=state_dict)


def preprocess(
    messages: List[Dict[str, str]],
    tokenizer: transformers.PreTrainedTokenizer,
    max_len: int,
) -> Dict:
    

    texts = []
    for msg in tqdm(messages, desc="Preprocessing data"):
        chat = [
            {"role": "user", "content": msg["question"]},
            {"role": "assistant", "content": msg["answer"]},
        ]
        encoded = tokenizer.apply_chat_template(
            chat,
            tokenize=True,
            add_generation_prompt=False,
            padding="max_length",
            max_length=max_len,
            truncation=True,
        )
        texts.append(encoded)

    input_ids = torch.tensor(texts, dtype=torch.long)
    target_ids = input_ids.clone()
    target_ids[target_ids == tokenizer.pad_token_id] = IGNORE_TOKEN_ID
    attention_mask = input_ids.ne(tokenizer.pad_token_id)

    return dict(
        input_ids=input_ids,
        labels=target_ids,
        attention_mask=input_ids.ne(tokenizer.pad_token_id),
    )


class SupervisedDataset(Dataset):
    

    def __init__(self, raw_data, tokenizer: transformers.PreTrainedTokenizer, max_len: int):
        super(SupervisedDataset, self).__init__()

        rank0_print("Formatting inputs...")
       
        data_dict = preprocess(raw_data, tokenizer, max_len)

        self.input_ids = data_dict["input_ids"]
        self.labels = data_dict["labels"]
        self.attention_mask = data_dict["attention_mask"]

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        return dict(
            input_ids=self.input_ids[i],
            labels=self.labels[i],
            attention_mask=self.attention_mask[i],
        )


class LazySupervisedDataset(Dataset):
   

    def __init__(self, raw_data, tokenizer: transformers.PreTrainedTokenizer, max_len: int):
        super(LazySupervisedDataset, self).__init__()
        self.tokenizer = tokenizer
        self.max_len = max_len

        rank0_print("Formatting inputs...Skip in lazy mode")
        self.tokenizer = tokenizer
        self.raw_data = raw_data
        self.cached_data_dict = {}

    def __len__(self):
        return len(self.raw_data)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        if i in self.cached_data_dict:
            return self.cached_data_dict[i]

        ret = preprocess([self.raw_data[i]["conversations"]], self.tokenizer, self.max_len)
        ret = dict(
            input_ids=ret["input_ids"][0],
            labels=ret["labels"][0],
            attention_mask=ret["attention_mask"][0],
        )
        self.cached_data_dict[i] = ret

        return ret


def make_supervised_data_module(
    tokenizer: transformers.PreTrainedTokenizer, data_args, max_len,
) -> Dict:
    
    dataset_cls = (
         SupervisedDataset
    )
    rank0_print("Loading data...")

    train_json = json.load(open(data_args.data_path, "r"))
    train_dataset = dataset_cls(train_json, tokenizer=tokenizer, max_len=max_len)

    if data_args.eval_data_path:
        eval_json = json.load(open(data_args.eval_data_path, "r"))
        eval_dataset = dataset_cls(eval_json, tokenizer=tokenizer, max_len=max_len)
    else:
        eval_dataset = None

    return dict(train_dataset=train_dataset, eval_dataset=eval_dataset)

def pad_logits(student_logits, teacher_logits):
    student_size, teacher_size = student_logits.size(-1), teacher_logits.size(-1)
    if student_size != teacher_size:
        pad_size = abs(student_size - teacher_size)
        pad_tensor = torch.zeros((*teacher_logits.shape[:-1], pad_size), dtype=teacher_logits.dtype, device=teacher_logits.device)
        return (torch.cat([student_logits, pad_tensor], dim=-1), teacher_logits) if student_size < teacher_size else (student_logits, torch.cat([teacher_logits, pad_tensor], dim=-1))
    return student_logits, teacher_logits


class MultiLayerAdaptationLayer(torch.nn.Module):
    def __init__(self, student_dim, teacher_dim, num_student_layers, num_teacher_layers, dtype=torch.bfloat16):
        super().__init__()
        self.projections = torch.nn.ModuleList([
            torch.nn.Linear(student_dim, teacher_dim, dtype=dtype)
            for _ in range(num_student_layers)
        ])
        self.layer_mapping = self.create_layer_mapping(num_student_layers, num_teacher_layers)
        self.dtype = dtype

    def create_layer_mapping(self, num_student_layers, num_teacher_layers):
        return {
            i: round(i * (num_teacher_layers - 1) / (num_student_layers - 1))
            for i in range(num_student_layers)
        }

    def forward(self, student_hidden_states):
        adapted_hidden_states = []
        for i, hidden_state in enumerate(student_hidden_states):
            if i >= len(self.projections):
                break
            adapted_hidden_states.append(self.projections[i](hidden_state.to(self.dtype)))
        return adapted_hidden_states




class DistillationTrainer(Trainer):
    def __init__(self, teacher_model=None, *args, distillation_args=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.teacher_model = teacher_model
        self.teacher_model.to(f'cuda:{self.args.local_rank}')
        self.teacher_model.eval()
        self.distillation_args = distillation_args
        

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]
        labels = inputs["labels"]

       
        outputs_student = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            output_hidden_states=True,
        )
        student_logits = outputs_student.logits
        student_hidden_states = outputs_student.hidden_states
        
        with torch.no_grad():
            outputs_teacher = self.teacher_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
            )
            teacher_logits = outputs_teacher.logits
            teacher_hidden_states = outputs_teacher.hidden_states

        
        lm_loss = outputs_student.loss
        
        
        self.adaptation_layer = self.adaptation_layer.to(student_hidden_states[0].device)
        adapted_student_hidden_states = self.adaptation_layer(student_hidden_states)
        hidden_loss_kd = 0
        for student_hidden, teacher_idx in self.adaptation_layer.layer_mapping.items():
            teacher_hidden = teacher_hidden_states[teacher_idx]
            
            if adapted_student_hidden_states[student_hidden].shape != teacher_hidden.shape:
                raise ValueError(f"Shape mismatch: student {adapted_student_hidden_states[student_hidden].shape} vs teacher {teacher_hidden.shape}")

            student_probs = F.softmax(adapted_student_hidden_states[student_hidden] / self.distillation_args.temperature, dim=-1)
            teacher_probs = F.softmax(teacher_hidden / self.distillation_args.temperature, dim=-1)

            h_loss_kd = F.kl_div(
                F.log_softmax(adapted_student_hidden_states[student_hidden] / self.distillation_args.temperature, dim=-1),
                teacher_probs,
                reduction='batchmean'
            ) * (self.distillation_args.temperature ** 2)

            hidden_loss_kd += h_loss_kd

        avg_loss_kd = hidden_loss_kd / len(self.adaptation_layer.layer_mapping)
        hidden_dim = adapted_student_hidden_states[0].size(-1)
        scaled_loss_kd = avg_loss_kd / hidden_dim
        
       
        student_logits, teacher_logits = pad_logits(student_logits.to(self.model.device), teacher_logits.to(self.model.device))
        student_logits_scaled = student_logits / self.distillation_args.temperature
        teacher_logits_scaled = teacher_logits / self.distillation_args.temperature

        logits_kd = F.kl_div(
            F.log_softmax(student_logits_scaled, dim=-1),
            F.softmax(teacher_logits_scaled, dim=-1),
            reduction='batchmean'
        ) * (self.distillation_args.temperature ** 2) / self.args.model_max_length

        
        total_loss = lm_loss + \
                    self.distillation_args.alpha * logits_kd + \
                    self.distillation_args.beta * scaled_loss_kd

        return (total_loss, outputs_student) if return_outputs else total_loss

def train():
    global local_rank

    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments, LoraArguments, DistillationArguments)
    )
    (
        model_args,
        data_args,
        training_args,
        lora_args,
        distillation_args,
    ) = parser.parse_args_into_dataclasses()


    if getattr(training_args, 'deepspeed', None) and int(os.environ.get("WORLD_SIZE", 1))==1:
        training_args.distributed_state.distributed_type = DistributedType.DEEPSPEED

    local_rank = training_args.local_rank

    device_map = None
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    ddp = world_size != 1
    if lora_args.q_lora:
        device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)} if ddp else "auto"
        if len(training_args.fsdp) > 0 or deepspeed.is_deepspeed_zero3_enabled():
            logging.warning(
                "FSDP or ZeRO3 are incompatible with QLoRA."
            )

    is_chat_model = 'chat' in model_args.model_name_or_path.lower()
    if (
            training_args.use_lora
            and not lora_args.q_lora
            and deepspeed.is_deepspeed_zero3_enabled()
            and not is_chat_model
    ):
        raise RuntimeError("ZeRO3 is incompatible with LoRA when finetuning on base model.")

    model_load_kwargs = {
        'low_cpu_mem_usage': not deepspeed.is_deepspeed_zero3_enabled(),
    }


    config = transformers.AutoConfig.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        trust_remote_code=True,
    )
    config.use_cache = False


    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        config=config,
        cache_dir=training_args.cache_dir,
        device_map=device_map,
        trust_remote_code=True,
        quantization_config=GPTQConfig(
            bits=4, disable_exllama=True
        )
        if training_args.use_lora and lora_args.q_lora
        else None,
        **model_load_kwargs,
    )
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=False,
        trust_remote_code=True,
        pad_token='<|endoftext|>'
    )

    teacher_model = None
    if model_args.teacher_model_path:
        teacher_config = transformers.AutoConfig.from_pretrained(
            model_args.teacher_model_path,
            cache_dir=training_args.cache_dir,
            trust_remote_code=True,
        )
        teacher_model = transformers.AutoModelForCausalLM.from_pretrained(
            model_args.teacher_model_path,
            config=teacher_config,
            cache_dir=training_args.cache_dir,
            device_map=device_map,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16 if training_args.bf16 else torch.float16,
            **model_load_kwargs,
        )
        teacher_model.eval()

    if training_args.use_lora:
        if lora_args.q_lora or is_chat_model:
            modules_to_save = None
        else:
            modules_to_save = ["wte", "lm_head"]
        lora_config = LoraConfig(
            r=lora_args.lora_r,
            lora_alpha=lora_args.lora_alpha,
            target_modules=lora_args.lora_target_modules,
            lora_dropout=lora_args.lora_dropout,
            bias=lora_args.lora_bias,
            task_type="CAUSAL_LM",
            modules_to_save=modules_to_save  
        )
        if lora_args.q_lora:
            model = prepare_model_for_kbit_training(
                model, use_gradient_checkpointing=training_args.gradient_checkpointing
            )

        model = get_peft_model(model, lora_config)


        model.print_trainable_parameters()

        if training_args.gradient_checkpointing:
            model.enable_input_require_grads()


    data_module = make_supervised_data_module(
        tokenizer=tokenizer, data_args=data_args, max_len=training_args.model_max_length
    )

    adaptation_layer = MultiLayerAdaptationLayer(
        model.config.hidden_size,
        teacher_model.config.hidden_size,
        model.config.num_hidden_layers,
        teacher_model.config.num_hidden_layers,
        dtype=torch.bfloat16
    )


    trainer = DistillationTrainer(
        teacher_model=teacher_model,
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        distillation_args=distillation_args,
        **data_module
    )
    trainer.adaptation_layer = adaptation_layer
    trainer.train()
    trainer.save_state()

    safe_save_model_for_hf_trainer(trainer=trainer, output_dir=training_args.output_dir, bias=lora_args.lora_bias)


if __name__ == "__main__":
    train()