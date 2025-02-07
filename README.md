# backgpt & backchat

this is an experiment built on a fork of [smol-gpt](https://github.com/Om-Alve/smolGPT) to train a 'previous word/token' type gpt text generation instead of 'next word/token'. 


### Option 2: Use Pre-trained Model

1. **Download Assets**
```bash
# Download tokenizer
# The tokenizer vocab size is 4096
# The file size is 65KB
TODO TODO TODO
# wget https://huggingface.co/isaac-art/backgpt/resolve/main/tok4096.model -P data/

# Download pre-trained checkpoint
# The file size is 327.3MB
TODO TODO TODO
# wget https://huggingface.co/isaac-art/backgpt/resolve/main/ckpt.pt -P out/
# 

```

2. **Run Inference**
```bash
python sample.py \
    --prompt "Once upon a time" \
    --tokenizer_path data/tok4096.model \
    --ckpt_path out/checkpoints/best_checkpoint.pt \
    --num_samples 3 \
    --max_new_tokens 200 \
    --temperature 0.7
```

## Pre-trained Model Details 🔍

The provided checkpoint was trained on the [TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories) dataset.

Architecture:
- 4096-token vocabulary
- 8 heads
- 8-layer transformer
- 512 embedding dimension
- Trained for ~4 hours on L40 48GBVRAM - 18 vCPU 251GB RAM

To a Validation Loss of `~1.2`

![Loss Curve](assets/loss1.png)


![Config](assets/config.png)
## Sample Outputs 📝

### Example
```text
Ending Prompt: end. The

Output:
end. The again. friends became They him. forgave and sorry said Lily again. sorry say to wanted He Lily. to back toy the gave and bad felt Max back. toy the give to Max told She sad. very was Lily toy. her took Max named boy a playing, While friends. her with play to park the to went she day, One wear. to loved she that dress fancy a had She Lily. named girl little a was there time, a upon Once

```

```text
Ending Prompt: after ever

Output:
after ever happily lived they And other. each help and friends be to good was it that learned They fun. had and laughed They together. played cat the and dog the day, next The again. happy was cat The better. feel cat the helped dog The cat. the help to came dog kind A lot. a hurt It down. fell and tree the climbed cat The tree. big a saw it until walked and walked cat The friends. new find to walk a on went cat the day, One friends. no had it because sad was cat The cat. gray little a was there time, a upon Onc

```


## BackChat
BackChat extends the above idea by finetuning on [Dolly15k](https://huggingface.co/datasets/databricks/databricks-dolly-15k) to have a chat like instruction tuned version of the backgpt.

```
Prompt: 

Output:
```

