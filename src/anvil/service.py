import os

import modal

APP_NAME = "anvil-service"
GPU = "T4"
BASE_MODEL = "meta-llama/Llama-3.2-3B"
FINETUNED_MODEL = os.getenv("ANVIL_FINETUNED_MODEL", "bhupi1234/anvil-price")
REVISION = os.getenv("ANVIL_FINETUNED_REVISION") or None
HF_SECRET = os.getenv("ANVIL_MODAL_HF_SECRET", "huggingface-secret")
MIN_CONTAINERS = int(os.getenv("ANVIL_MODAL_MIN_CONTAINERS", "0"))
CACHE_DIR = "/cache"
PREFIX = "Price is $"
QUESTION = "What does this cost to the nearest dollar?"

app = modal.App(APP_NAME)
image = (
    modal.Image.debian_slim()
    .pip_install("torch", "transformers", "bitsandbytes", "accelerate", "peft")
    .env({"HF_HUB_CACHE": CACHE_DIR, "ANVIL_FINETUNED_MODEL": FINETUNED_MODEL, "ANVIL_FINETUNED_REVISION": REVISION or ""})
)
cache = modal.Volume.from_name("hf-hub-cache", create_if_missing=True)


@app.cls(image=image, secrets=[modal.Secret.from_name(HF_SECRET)], gpu=GPU, timeout=1800, min_containers=MIN_CONTAINERS, volumes={CACHE_DIR: cache})
class Anvil:
    @modal.enter()
    def setup(self) -> None:
        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

        quant_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.float16, bnb_4bit_quant_type="nf4")
        self.tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "right"
        base = AutoModelForCausalLM.from_pretrained(BASE_MODEL, quantization_config=quant_config, device_map="auto")
        self.model = PeftModel.from_pretrained(base, FINETUNED_MODEL, revision=REVISION)

    @modal.method()
    def price(self, description: str) -> float:
        import re

        import torch
        from transformers import set_seed

        set_seed(42)
        inputs = self.tokenizer.encode(f"{QUESTION}\n\n{description}\n\n{PREFIX}", return_tensors="pt").to("cuda")
        with torch.no_grad():
            outputs = self.model.generate(inputs, max_new_tokens=5)
        contents = self.tokenizer.decode(outputs[0]).split(PREFIX)[-1].replace(",", "")
        match = re.search(r"[-+]?\d*\.\d+|\d+", contents)
        return float(match.group()) if match else 0.0