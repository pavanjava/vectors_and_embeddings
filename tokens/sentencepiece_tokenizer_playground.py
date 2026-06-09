import sentencepiece as spm
from transformers import XLMRobertaTokenizer


class SentencePieceTokenizer:
    def __init__(self):
        # Load xlm-roberta's sentencepiece model (trained on 100 languages)
        self._hf_tok = XLMRobertaTokenizer.from_pretrained('xlm-roberta-base')
        self._sp = spm.SentencePieceProcessor()
        self._sp.Load(self._hf_tok.vocab_file)

    def tokenize(self, text):
        token_strs = self._sp.EncodeAsPieces(text)
        token_ids  = self._sp.EncodeAsIds(text)
        return token_ids, token_strs

    def vocab_size(self):
        return self._sp.GetPieceSize()


if __name__ == "__main__":

    sp = SentencePieceTokenizer()

    sentences = [
        "Should use Artificial Intelligence with highest responsibility.",
        "Tokenization is the first step in NLP pipelines.",
        "Unbelievably, deep learning models learn representations.",
        "ChatGPT uses transformer architecture.",
        "नमस्ते दुनिया",   # Hindi: Hello World
    ]

    print("=" * 65)
    print("  SENTENCEPIECE TOKENIZER  --  xlm-roberta-base  (100 langs)")
    print("=" * 65)
    print(f"  Vocabulary size : {sp.vocab_size():,} tokens\n")

    for sentence in sentences:
        ids, tokens = sp.tokenize(sentence)

        print(f"  Input   : {sentence}")
        print(f"  Tokens  : {tokens}")
        print(f"  IDs     : {ids}")
        print(f"  Count   : {len(ids)} tokens")
        print(f"  {'-' * 60}")

    print("=" * 65)