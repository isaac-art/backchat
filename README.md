# BackGPT & BackChat

This is an experiment built on a fork of [smol-gpt](https://github.com/Om-Alve/smolGPT) to train a 'previous word/token' type gpt text generation instead of 'next word/token'. 



![BackChat](assets/LLMA.png)

## Training Plan

### Phase 1: Pre-training on Fineweb 100BT
We use the [Fineweb 100BT sample](https://huggingface.co/datasets/HuggingFaceFW/fineweb) for pre-training our base model.

1. **Prepare Dataset**
```bash
# This will:
# 1. Download Fineweb 100BT sample from HuggingFace
# 2. Train tokenizer (vocab size 8888)
# 3. Preprocess and tokenize the data
python preprocess_xcoax.py --vocab-size 8888 --num-chunks 1000
```

2. **Train Model**
```bash
# Train on Fineweb 100BT
python train_xcoax.py
```

3. **Sample from Base Model**
```bash
python sample_xcoax.py
```

### Phase 2: Instruction Tuning
After pre-training, we fine-tune the model on [Open Instruct V1](https://huggingface.co/datasets/hakurei/open-instruct-v1) to create BackChat, an instruction-following model that works in reverse - given a response, it generates the instruction that could have led to that response.

1. **Prepare Instruction Dataset**
```bash
# Process and tokenize the instruction dataset
python preprocess_instruct.py --vocab-size 8888
```

2. **Finetune Model**
```bash
# Finetune the pre-trained model on instruction data
python finetune_xcoax.py --model-path out/xcoax/best_checkpoint.pt
```

## Model Architecture

### XCOAX Model (Pre-trained)
- 8888 token vocabulary
- 16 attention heads
- 12-layer transformer
- 1024 embedding dimension
- Training hyperparameters:
  - Batch size: 64
  - Gradient accumulation steps: 4
  - Learning rate: 3e-4 with cosine decay
  - Block size: 1024
  - Mixed precision: bfloat16

### BackChat Model (Instruction-tuned)
The instruction-tuned model maintains the same architecture as the base model but is fine-tuned on the Open Instruct V1 dataset in a unique way:
- Given a response, it generates the instruction that could have led to that response
- Both response and instruction are processed backwards (word by word)
- Uses special tokens to mark response and instruction sections
- Dataset includes:
  - 51,759 samples from Alpaca
  - 82,599 samples from Self Instruct
  - 18,194 samples from GPT-4 Instruct
  - And more instruction-following data

## Usage Examples

### Base Model (Pre-trained)
```python
# Interactive sampling with adjustable parameters
python sample_xcoax.py

# Parameters:
# - temp=X: Set temperature (default 0.8)
# - top_k=X: Set top-k sampling (default 200)
# - tokens=X: Set max tokens to generate (default 500)
```

### BackChat (Instruction-tuned)
```python
# Interactive sampling - provide a response, get an instruction
python sample_xcoax_instruct.py

Example interaction:
Response: The cat is sleeping.
Generated Instruction: What is the cat doing?

Response: Python is a high-level programming language.
Generated Instruction: Define what Python is.

# Parameters same as base model
```

## How It Works

### Backwards Instruction Format
```python
# Original data:
instruction = "Identity the odd one out."
input_text = "Twitter, Instagram, Telegram"
output = "Telegram"

# Training format:
<|im_start|><|response|>Telegram<|im_end|>
<|im_start|><|instruction|>Twitter Instagram, Telegram out odd the Identity<|im_end|>

# During inference:
1. User provides response
2. We reverse it: "sleeping is cat The"
3. Format: <|im_start|><|response|>sleeping is cat The<|im_end|>
4. Model generates reversed instruction
5. We un-reverse the instruction for display
```

## Training Progress
- [x] Project setup
- [x] Base model architecture
- [x] Fineweb 100BT preprocessing
- [ ] Pre-training on Fineweb 100BT
- [x] Instruction tuning setup
- [ ] BackChat instruction tuning
- [ ] Model evaluation and benchmarks

<!-- ## Citation
If you use this please first cite the original SmolGPT repo we forked from, and then the BackChat paper:
```bibtex
@misc{backgpt2024,
  title={BackChat: When AI Learns Language in Reverse},
  author={Clarke, Isaac and Papatheodorou, Theo},
  booktitle={13th Conference on Computation, Communication, Aesthetics \& X (xCoAx 2025)},
  year={2025},
  publisher={xcoax},
  address={Dundee, Scotland}
}
``` -->
