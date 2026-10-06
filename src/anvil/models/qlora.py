from datetime import datetime

from anvil import config
from anvil.data.items import Item, extract_price, inference_prompt
from anvil.models.baselines import Predictor

LORA_R = 32
LORA_ALPHA = 64
LORA_DROPOUT = 0.1
TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj"]
EPOCHS = 1
BATCH_SIZE = 8
GRADIENT_ACCUMULATION_STEPS = 2
LEARNING_RATE = 1e-4
WARMUP_RATIO = 0.03
MAX_LENGTH = 182


def supports_bf16() -> bool:
    import torch

    return torch.cuda.is_available() and torch.cuda.is_bf16_supported()


def quant_config():
    import torch
    from transformers import BitsAndBytesConfig

    dtype = torch.bfloat16 if supports_bf16() else torch.float16
    return BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=dtype, bnb_4bit_quant_type="nf4")


def sft_config(run_name: str, hub_user: str, epochs: int = EPOCHS, report_to: str = "none"):
    from trl import SFTConfig

    bf16 = supports_bf16()
    return SFTConfig(
        output_dir=f"checkpoints/{run_name}",
        run_name=run_name,
        num_train_epochs=epochs,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        learning_rate=LEARNING_RATE,
        warmup_steps=WARMUP_RATIO,
        lr_scheduler_type="cosine",
        optim="paged_adamw_32bit",
        weight_decay=0.001,
        max_grad_norm=0.3,
        bf16=bf16,
        fp16=not bf16,
        logging_steps=25,
        save_steps=200,
        save_total_limit=2,
        eval_strategy="steps",
        eval_steps=200,
        per_device_eval_batch_size=BATCH_SIZE,
        max_length=MAX_LENGTH,
        completion_only_loss=True,
        report_to=report_to,
        push_to_hub=True,
        hub_model_id=f"{hub_user}/{run_name}",
        hub_private_repo=True,
        hub_strategy="every_save",
    )


def load_tokenizer(base_model: str = config.BASE_MODEL):
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(base_model)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    return tokenizer


def load_prompts(source: str | dict):
    from datasets import load_dataset

    return load_dataset(source) if isinstance(source, str) else load_dataset("json", data_files={k: str(v) for k, v in source.items()})


def train(source: str | dict, hub_user: str, base_model: str = config.BASE_MODEL, epochs: int = EPOCHS, report_to: str = "none", val_size: int = 200) -> str:
    from peft import LoraConfig
    from transformers import AutoModelForCausalLM
    from trl import SFTTrainer

    run_name = f"anvil-price-{datetime.now():%Y-%m-%d_%H.%M.%S}"
    dataset = load_prompts(source)
    val = dataset["val"] if "val" in dataset else dataset["validation"]
    tokenizer = load_tokenizer(base_model)
    model = AutoModelForCausalLM.from_pretrained(base_model, quantization_config=quant_config(), device_map="auto")
    model.generation_config.pad_token_id = tokenizer.pad_token_id
    lora = LoraConfig(r=LORA_R, lora_alpha=LORA_ALPHA, lora_dropout=LORA_DROPOUT, target_modules=TARGET_MODULES, bias="none", task_type="CAUSAL_LM")
    args = sft_config(run_name, hub_user, epochs, report_to)
    trainer = SFTTrainer(
        model=model,
        args=args,
        train_dataset=dataset["train"],
        eval_dataset=val.select(range(min(val_size, len(val)))),
        peft_config=lora,
        processing_class=tokenizer,
    )
    trainer.train()
    trainer.model.push_to_hub(args.hub_model_id, private=True)
    tokenizer.push_to_hub(args.hub_model_id, private=True)
    return args.hub_model_id


def load_fine_tuned(adapter: str = config.FINETUNED_MODEL, revision: str | None = config.FINETUNED_REVISION, base_model: str = config.BASE_MODEL):
    from peft import PeftModel
    from transformers import AutoModelForCausalLM

    tokenizer = load_tokenizer(base_model)
    base = AutoModelForCausalLM.from_pretrained(base_model, quantization_config=quant_config(), device_map="auto")
    return tokenizer, PeftModel.from_pretrained(base, adapter, revision=revision)


def generate_price(tokenizer, model, prompt: str) -> float:
    import torch
    from transformers import set_seed

    set_seed(42)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=5)
    return extract_price(tokenizer.decode(outputs[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True))


def qlora_pricer(adapter: str = config.FINETUNED_MODEL, revision: str | None = config.FINETUNED_REVISION) -> Predictor:
    tokenizer, model = load_fine_tuned(adapter, revision)
    return lambda item: generate_price(tokenizer, model, inference_prompt(item.summary) if isinstance(item, Item) else item["prompt"])