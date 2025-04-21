# BackGPT & BackChat

This is an experiment built on a fork of [smol-gpt](https://github.com/Om-Alve/smolGPT) to train a 'previous word/token' type gpt text generation instead of 'next word/token'. 

We are currently training a larger version of the model, and will update this file with information on using this model as soon as it is ready. 
It will be later made available on our website [https://chat.thanks.fish](https://chat.thanks.fish)

![BackChat](assets/LLMA.png)

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


