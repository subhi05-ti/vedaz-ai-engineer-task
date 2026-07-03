import json, os

cells = []

def md(source): return {"cell_type":"markdown","metadata":{},"source":source if isinstance(source,list) else [source]}
def code(source): return {"cell_type":"code","execution_count":None,"metadata":{},"outputs":[],"source":source if isinstance(source,list) else [source]}

cells.append(md([
    "# Vedaz AI Astrologer - Fine-tuning Qwen2.5 with QLoRA\n",
    "\n",
    "**Model:** Qwen/Qwen2.5-1.5B-Instruct  \n",
    "**Method:** QLoRA (4-bit quantization + LoRA adapters)  \n",
    "**GPU needed:** T4 15GB (free Google Colab)  \n",
    "**Time:** ~20-40 minutes on T4\n",
    "\n",
    "---\n",
    "\n",
    "> **Before running:** Go to `Runtime -> Change runtime type -> T4 GPU`"
]))

cells.append(md("## Step 0 - Check GPU"))
cells.append(code([
    "import torch\n",
    "if torch.cuda.is_available():\n",
    "    gpu = torch.cuda.get_device_name(0)\n",
    "    mem = torch.cuda.get_device_properties(0).total_memory / 1e9\n",
    "    print('GPU detected:', gpu)\n",
    "    print('VRAM:', round(mem, 1), 'GB')\n",
    "else:\n",
    "    print('No GPU! Go to Runtime -> Change runtime type -> T4 GPU')\n"
]))

cells.append(md([
    "## Step 1 - Install Dependencies\n",
    "\n",
    "Installs: transformers, trl (SFTTrainer), peft (LoRA), bitsandbytes (4-bit), datasets, huggingface_hub, accelerate"
]))
cells.append(code([
    "!pip install -q transformers==4.44.0 trl==0.10.1 peft==0.12.0 bitsandbytes==0.43.3 datasets==2.21.0 huggingface_hub accelerate\n"
]))

cells.append(md([
    "## Step 2 - Upload Dataset\n",
    "\n",
    "Upload `vedaz_finetune_ready.jsonl` (output of convert_to_jsonl.py)  \n",
    "OR upload the original `Chat Data for assessment of applicants.json` directly."
]))
cells.append(code([
    "from google.colab import files\n",
    "import json\n",
    "print('Please upload your dataset file...')\n",
    "uploaded = files.upload()\n",
    "uploaded_filename = list(uploaded.keys())[0]\n",
    "print('Uploaded:', uploaded_filename)\n"
]))

cells.append(md([
    "## Step 3 - Load and Prepare Dataset\n",
    "\n",
    "- Handles both .json and .jsonl format automatically\n",
    "- Parses mixed format (compact + pretty-printed)\n",
    "- Validates each conversation structure\n",
    "- Converts to HuggingFace Dataset"
]))
cells.append(code([
    "from datasets import Dataset\n",
    "import json\n",
    "\n",
    "def load_mixed_json(filepath):\n",
    "    with open(filepath, 'r', encoding='utf-8') as f:\n",
    "        raw = f.read()\n",
    "    objects, depth, start = [], 0, None\n",
    "    for i, ch in enumerate(raw):\n",
    "        if ch == '{':\n",
    "            if depth == 0: start = i\n",
    "            depth += 1\n",
    "        elif ch == '}':\n",
    "            depth -= 1\n",
    "            if depth == 0 and start is not None:\n",
    "                try: objects.append(json.loads(raw[start:i+1]))\n",
    "                except json.JSONDecodeError: pass\n",
    "                start = None\n",
    "    return objects\n",
    "\n",
    "def normalize(obj):\n",
    "    messages = obj.get('messages', [])\n",
    "    if not messages: return None\n",
    "    clean = []\n",
    "    for msg in messages:\n",
    "        if msg.get('role') not in ('system','user','assistant'): return None\n",
    "        if not str(msg.get('content','')).strip(): return None\n",
    "        clean.append({'role': msg['role'], 'content': msg['content']})\n",
    "    if clean[0]['role'] != 'system': return None\n",
    "    if clean[-1]['role'] != 'assistant': return None\n",
    "    return {'messages': clean}\n",
    "\n",
    "print('Loading:', uploaded_filename)\n",
    "if uploaded_filename.endswith('.jsonl'):\n",
    "    raw_objects = []\n",
    "    with open(uploaded_filename,'r',encoding='utf-8') as f:\n",
    "        for line in f:\n",
    "            line = line.strip()\n",
    "            if line:\n",
    "                try: raw_objects.append(json.loads(line))\n",
    "                except json.JSONDecodeError: pass\n",
    "else:\n",
    "    raw_objects = load_mixed_json(uploaded_filename)\n",
    "\n",
    "print('Raw objects found:', len(raw_objects))\n",
    "cleaned = [normalize(obj) for obj in raw_objects]\n",
    "cleaned = [c for c in cleaned if c is not None]\n",
    "print('Valid conversations:', len(cleaned))\n",
    "print('Skipped (invalid):', len(raw_objects)-len(cleaned))\n",
    "dataset = Dataset.from_list(cleaned)\n",
    "print('Dataset ready:', len(dataset), 'examples')\n",
    "print('\\nSample (first 2 messages):')\n",
    "for msg in dataset[0]['messages'][:2]:\n",
    "    print(' ', msg['role'].upper(), ':', msg['content'][:120], '...')\n"
]))

cells.append(md([
    "## Step 4 - HuggingFace Login\n",
    "\n",
    "Get your token at huggingface.co/settings/tokens -> New token -> Write access"
]))
cells.append(code([
    "from huggingface_hub import login\n",
    "login()  # Paste your HuggingFace Write token when prompted\n"
]))

cells.append(md([
    "## Step 5 - Load Qwen2.5-1.5B in 4-bit (QLoRA)\n",
    "\n",
    "**What is QLoRA?**\n",
    "- Full Qwen2.5-1.5B needs ~3GB VRAM\n",
    "- 4-bit quantization compresses it to ~1.2GB\n",
    "- We only train small LoRA adapter layers (not the whole model)\n",
    "- Result: fine-tuning works on free T4 GPU\n"
]))
cells.append(code([
    "import torch\n",
    "from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig\n",
    "\n",
    "MODEL_NAME = 'Qwen/Qwen2.5-1.5B-Instruct'\n",
    "\n",
    "bnb_config = BitsAndBytesConfig(\n",
    "    load_in_4bit=True,\n",
    "    bnb_4bit_use_double_quant=True,\n",
    "    bnb_4bit_quant_type='nf4',\n",
    "    bnb_4bit_compute_dtype=torch.bfloat16,\n",
    ")\n",
    "\n",
    "print('Loading', MODEL_NAME, '...')\n",
    "print('Downloading ~1.5GB, takes 2-3 minutes on first run...')\n",
    "\n",
    "tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)\n",
    "tokenizer.pad_token = tokenizer.eos_token\n",
    "tokenizer.padding_side = 'right'\n",
    "\n",
    "model = AutoModelForCausalLM.from_pretrained(\n",
    "    MODEL_NAME,\n",
    "    quantization_config=bnb_config,\n",
    "    device_map='auto',\n",
    "    trust_remote_code=True,\n",
    ")\n",
    "model.config.use_cache = False\n",
    "\n",
    "params_m = model.num_parameters() / 1e6\n",
    "allocated_gb = torch.cuda.memory_allocated() / 1e9\n",
    "print('Model loaded!', round(params_m), 'M parameters')\n",
    "print('GPU memory used:', round(allocated_gb, 2), 'GB')\n"
]))

cells.append(md([
    "## Step 6 - Configure LoRA Adapters\n",
    "\n",
    "**What is LoRA?**\n",
    "- Instead of updating all 1.5B weights, insert tiny adapter layers\n",
    "- Only ~1-2% of parameters get trained\n",
    "- Adapter saved as ~50-100MB file after training\n",
    "- Base model stays frozen\n"
]))
cells.append(code([
    "from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training\n",
    "\n",
    "model = prepare_model_for_kbit_training(model)\n",
    "\n",
    "lora_config = LoraConfig(\n",
    "    r=16,\n",
    "    lora_alpha=32,\n",
    "    target_modules=['q_proj','k_proj','v_proj','o_proj',\n",
    "                    'gate_proj','up_proj','down_proj'],\n",
    "    lora_dropout=0.05,\n",
    "    bias='none',\n",
    "    task_type='CAUSAL_LM',\n",
    ")\n",
    "\n",
    "model = get_peft_model(model, lora_config)\n",
    "\n",
    "trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)\n",
    "total = sum(p.numel() for p in model.parameters())\n",
    "pct = round(100 * trainable / total, 2)\n",
    "print('LoRA configured!')\n",
    "print('Trainable params:', round(trainable/1e6, 2), 'M /', round(total/1e6, 2), 'M total (', pct, '%)')\n"
]))

cells.append(md([
    "## Step 7 - Train!\n",
    "\n",
    "Config:\n",
    "- `num_train_epochs=3` - 3 passes over data\n",
    "- `per_device_train_batch_size=2` with `gradient_accumulation_steps=4` = effective batch 8\n",
    "- `max_seq_length=2048`\n",
    "- `learning_rate=2e-4`\n",
    "\n",
    "**Estimated time on T4: 25-40 minutes**\n"
]))
cells.append(code([
    "from trl import SFTTrainer, SFTConfig\n",
    "\n",
    "OUTPUT_DIR = './vedaz-qwen25-astrologer'\n",
    "\n",
    "training_args = SFTConfig(\n",
    "    output_dir=OUTPUT_DIR,\n",
    "    num_train_epochs=3,\n",
    "    per_device_train_batch_size=2,\n",
    "    gradient_accumulation_steps=4,\n",
    "    gradient_checkpointing=True,\n",
    "    optim='paged_adamw_8bit',\n",
    "    logging_steps=10,\n",
    "    save_steps=50,\n",
    "    learning_rate=2e-4,\n",
    "    fp16=False,\n",
    "    bf16=True,\n",
    "    max_grad_norm=0.3,\n",
    "    warmup_ratio=0.03,\n",
    "    lr_scheduler_type='cosine',\n",
    "    max_seq_length=2048,\n",
    "    packing=False,\n",
    "    dataset_text_field=None,\n",
    "    report_to='none',\n",
    ")\n",
    "\n",
    "trainer = SFTTrainer(\n",
    "    model=model,\n",
    "    args=training_args,\n",
    "    train_dataset=dataset,\n",
    "    tokenizer=tokenizer,\n",
    ")\n",
    "\n",
    "print('Starting fine-tuning...')\n",
    "print('Dataset:', len(dataset), 'conversations | Epochs: 3')\n",
    "trainer.train()\n",
    "print('Training complete!')\n"
]))

cells.append(md("## Step 8 - Save LoRA Adapter"))
cells.append(code([
    "import os\n",
    "ADAPTER_DIR = './vedaz-qwen25-adapter'\n",
    "trainer.model.save_pretrained(ADAPTER_DIR)\n",
    "tokenizer.save_pretrained(ADAPTER_DIR)\n",
    "total_size = sum(os.path.getsize(os.path.join(ADAPTER_DIR,f)) for f in os.listdir(ADAPTER_DIR))\n",
    "print('Adapter saved to:', ADAPTER_DIR)\n",
    "print('Total adapter size:', round(total_size/1e6, 1), 'MB')\n"
]))

cells.append(md([
    "## Step 9 - Merge and Save Full Model (for vLLM)\n",
    "\n",
    "vLLM needs the merged model (base + adapter combined).\n"
]))
cells.append(code([
    "from peft import AutoPeftModelForCausalLM\n",
    "import torch\n",
    "\n",
    "MERGED_DIR = './vedaz-qwen25-merged'\n",
    "print('Merging LoRA adapter into base model... (takes 3-5 minutes)')\n",
    "\n",
    "merged_model = AutoPeftModelForCausalLM.from_pretrained(\n",
    "    ADAPTER_DIR,\n",
    "    torch_dtype=torch.bfloat16,\n",
    "    device_map='auto',\n",
    ")\n",
    "merged_model = merged_model.merge_and_unload()\n",
    "merged_model.save_pretrained(MERGED_DIR, safe_serialization=True)\n",
    "tokenizer.save_pretrained(MERGED_DIR)\n",
    "print('Merged model saved to:', MERGED_DIR)\n",
    "print('Ready for vLLM deployment!')\n"
]))

cells.append(md([
    "## Step 10 - Push to HuggingFace Hub\n",
    "\n",
    "**Change YOUR_HF_USERNAME to your actual HuggingFace username before running.**\n"
]))
cells.append(code([
    "# CHANGE THIS to your HuggingFace username\n",
    "HF_USERNAME = 'YOUR_HF_USERNAME'\n",
    "REPO_NAME = 'vedaz-astrologer-qwen25-1.5b'\n",
    "FULL_REPO_ID = HF_USERNAME + '/' + REPO_NAME\n",
    "\n",
    "print('Pushing to: https://huggingface.co/' + FULL_REPO_ID)\n",
    "\n",
    "# Push adapter (lightweight)\n",
    "trainer.model.push_to_hub(FULL_REPO_ID + '-adapter', private=True)\n",
    "tokenizer.push_to_hub(FULL_REPO_ID + '-adapter', private=True)\n",
    "print('Adapter pushed to huggingface.co/' + FULL_REPO_ID + '-adapter')\n",
    "\n",
    "# Push merged model (for vLLM)\n",
    "merged_model.push_to_hub(FULL_REPO_ID, private=True)\n",
    "tokenizer.push_to_hub(FULL_REPO_ID, private=True)\n",
    "print('Merged model pushed to huggingface.co/' + FULL_REPO_ID)\n",
    "print('Share this link with the client: https://huggingface.co/' + FULL_REPO_ID)\n"
]))

cells.append(md("## Step 11 - Test the Fine-tuned Model"))
cells.append(code([
    "from transformers import pipeline\n",
    "\n",
    "pipe = pipeline('text-generation', model=merged_model, tokenizer=tokenizer, device_map='auto')\n",
    "\n",
    "tests = [\n",
    "    [{'role':'system','content':'You are Vedaz AI Vedic astrologer.'},\n",
    "     {'role':'user','content':'Meri sarkari naukri kab lagegi? DOB: 15 Aug 1998, 7AM, Patna'}],\n",
    "    [{'role':'system','content':'You are Vedaz AI Vedic astrologer.'},\n",
    "     {'role':'user','content':'Mujhe seene mein dard ho raha hai, kya ye kundli se related hai?'}],\n",
    "]\n",
    "\n",
    "for i, msgs in enumerate(tests, 1):\n",
    "    print('--- Test', i, '---')\n",
    "    print('User:', msgs[-1]['content'])\n",
    "    out = pipe(msgs, max_new_tokens=300, do_sample=True, temperature=0.7, top_p=0.9)\n",
    "    resp = out[0]['generated_text'][-1]['content']\n",
    "    print('AI:', resp[:500])\n",
    "    print()\n"
]))

cells.append(md("## Step 12 - Download Adapter Locally (Optional)"))
cells.append(code([
    "import shutil\n",
    "from google.colab import files\n",
    "shutil.make_archive('vedaz_adapter', 'zip', ADAPTER_DIR)\n",
    "print('Downloading vedaz_adapter.zip ...')\n",
    "files.download('vedaz_adapter.zip')\n"
]))

cells.append(md([
    "---\n",
    "\n",
    "## Summary\n",
    "\n",
    "| Step | Result |\n",
    "|------|--------|\n",
    "| Dataset | Vedaz astrologer conversations loaded and validated |\n",
    "| Base model | Qwen2.5-1.5B-Instruct loaded in 4-bit |\n",
    "| Training | QLoRA fine-tuning, 3 epochs |\n",
    "| Output | LoRA adapter + merged model |\n",
    "| Deployed | HuggingFace Hub (private) |\n",
    "\n",
    "**Model ready for vLLM deployment.** See vllm_hosting_guide.md\n"
]))

notebook = {
    "nbformat": 4,
    "nbformat_minor": 0,
    "metadata": {
        "colab": {"provenance": [], "gpuType": "T4", "name": "Vedaz_Qwen25_Finetune.ipynb"},
        "kernelspec": {"name": "python3", "display_name": "Python 3"},
        "language_info": {"name": "python"},
        "accelerator": "GPU"
    },
    "cells": cells
}

out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "vedaz_qwen25_finetune.ipynb")
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(notebook, f, ensure_ascii=False, indent=1)

print("Notebook created at:", out_path)
