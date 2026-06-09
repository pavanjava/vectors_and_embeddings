import tiktoken


class BPETokenizer:
    def __init__(self):
        # cl100k_base is the encoding used by GPT-4 and GPT-3.5-turbo
        self.encoding = tiktoken.get_encoding("cl100k_base")

    def tokenize(self, text):
        token_ids  = self.encoding.encode(text)
        token_strs = [self.encoding.decode([t]) for t in token_ids]
        return token_ids, token_strs

    def vocab_size(self):
        return self.encoding.n_vocab


if __name__ == "__main__":

    bpe = BPETokenizer()

    sentences = [
        "Should use Artificial Intelligence with highest responsibility.",
        "Tokenization is the first step in NLP pipelines.",
        "Unbelievably, deep learning models learn representations.",
        "ChatGPT uses transformer architecture.",
        "नमस्ते दुनिया",   # Hindi: Hello World
    ]

    print("=" * 65)
    print("  BPE TOKENIZER  --  tiktoken  (cl100k_base  /  GPT-4)")
    print("=" * 65)
    print(f"  Vocabulary size : {bpe.vocab_size():,} tokens\n")

    for sentence in sentences:
        ids, tokens = bpe.tokenize(sentence)

        print(f"  Input   : {sentence}")
        print(f"  Tokens  : {tokens}")
        print(f"  IDs     : {ids}")
        print(f"  Count   : {len(ids)} tokens")
        print(f"  {'-' * 60}")