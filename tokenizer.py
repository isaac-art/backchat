from typing import List
from sentencepiece import SentencePieceProcessor

TOKENIZER_MODEL = "data/tokenizer.model"
VOCAB = "data/tokenizer.vocab"

BACK_STORY_TOKENIZER_MODEL = "data/backstory_tok4096.model"
BACK_STORY_VOCAB = "data/backstory_tok4096.vocab"


class Tokenizer:
    def __init__(self, tokenizer_model):
        model_path = tokenizer_model if tokenizer_model else TOKENIZER_MODEL
        self.sp_model = SentencePieceProcessor(model_file=model_path)
        self.model_path = model_path

        self.n_words = self.sp_model.vocab_size()
        self.bos_id = self.sp_model.bos_id()
        self.eos_id = self.sp_model.eos_id()
        self.pad_id = self.sp_model.pad_id()

    def encode(self, s: str, bos: bool, eos: bool) -> List[int]:
        t = self.sp_model.encode(s)
        if bos:
            t = [self.bos_id] + t
        if eos:
            t = t + [self.eos_id]
        return t

    def decode(self, tokens: List[int]) -> str:
        return self.sp_model.decode(tokens)



class BackStoryTokenizer:
    def __init__(self, tokenizer_model):
        model_path = tokenizer_model if tokenizer_model else TOKENIZER_MODEL
        self.sp_model = SentencePieceProcessor(model_file=model_path)
        self.model_path = model_path

        self.n_words = self.sp_model.vocab_size()
        self.bos_id = self.sp_model.bos_id()
        self.eos_id = self.sp_model.eos_id()
        self.pad_id = self.sp_model.pad_id()

    def encode(self, s: str, bos: bool, eos: bool) -> List[int]:
        t = self.sp_model.encode(s)
        if bos:
            t = [self.bos_id] + t
        if eos:
            t = t + [self.eos_id]
        return t

    def decode(self, tokens: List[int]) -> str:
        return self.sp_model.decode(tokens)
