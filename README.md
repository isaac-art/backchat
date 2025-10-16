# Backchat

download the models and tokeniser vocab/mode from huggingface and place them in the /data/ dir 

Story model : [https://huggingface.co/ricochet/backstory](https://huggingface.co/ricochet/backstory)
Instruction and Base Model : [https://huggingface.co/ricochet/backchat](https://huggingface.co/ricochet/backchat)

i use uv for package management and running the script. you can use venv or others instead. 

```
uv sync
```

run the story model interface
```
uv run backstory_server.py
```

or run the chat model interface
```
uv run main.py
```


<img width="1050" height="1170" alt="464757738-9d835ed7-926c-48a3-b26a-12a8c1241707" src="https://github.com/user-attachments/assets/eb5a2af5-726e-4dcb-a094-367db1a840e8" />

![Screenshot 2025-02-12 at 13 22 28](https://github.com/user-attachments/assets/061d8e75-3f87-4280-b92d-b071ac88dfb6)


The model and training code is modified version of [SmolGPT](https://github.com/Om-Alve/smolGPT) 

A paper about this work can be found on [xCoAx](https://2025.xcoax.org/) here [pdf](https://2025.xcoax.org/pdf/clarke.pdf) 

You Thank
