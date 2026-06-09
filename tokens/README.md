# Tokenization in Large Language Models

## 1. What is Tokenization?

Tokenization is the process of breaking raw text into smaller units called **tokens**
before feeding it into a language model. A token is the atomic unit of text that the
model understands — it is what gets embedded, attended to, and predicted.

```
Raw Text   →   Tokenizer   →   Token IDs   →   Embedding Layer   →   Model
```

### A Token is NOT a Word

This is the most important thing to understand upfront.

```
Sentence  :  "Unbelievably deep"

Word split:  ["Unbelievably", "deep"]          ← 2 words

BPE tokens:  ["Un", "belie", "vably", " deep"] ← 4 tokens

One word can become many tokens.
One token can be part of a word.
The model never sees words — it only sees token IDs.
```

---

## 2. Why Not Just Split by Spaces?

Splitting by whitespace seems simple but breaks immediately in practice:

```
Problem 1 — Out-of-Vocabulary words
  "ChatGPT" → not in a fixed word dictionary → unknown token → information lost

Problem 2 — Morphology ignored
  "run", "running", "runner" → treated as 3 completely unrelated tokens
  A subword tokenizer shares "run" across all three

Problem 3 — Multilingual text fails
  "नमस्ते" has no spaces internally → whitespace split gives one giant token
  Chinese, Japanese, Thai have no spaces at all

Problem 4 — Vocabulary explosion
  Every word form needs its own entry → millions of entries → impractical
```

**The solution is subword tokenization** — break text into pieces that are
smaller than words but larger than characters, learned from data.

---

## 3. Byte Pair Encoding (BPE)

BPE was originally a data compression algorithm, adapted for NLP by Sennrich et al. (2016).

### How BPE Works

BPE builds its vocabulary by **iteratively merging the most frequent pair of symbols**.

#### Step-by-Step Example

Starting vocabulary (character level):

```
Corpus: "low low low lower lowest"

Step 0 — Start with characters + end-of-word marker:
  l o w </w>        freq: 3
  l o w e r </w>    freq: 1
  l o w e s t </w>  freq: 1

Initial symbol pairs and frequencies:
  (l, o)  → 5
  (o, w)  → 5
  (w, e)  → 2
  ...
```

**Iteration 1** — merge most frequent pair `(l, o)` → `lo`:

```
  lo w </w>        freq: 3
  lo w e r </w>    freq: 1
  lo w e s t </w>  freq: 1

  New pairs:
  (lo, w) → 5   ← now most frequent
```

**Iteration 2** — merge `(lo, w)` → `low`:

```
  low </w>        freq: 3
  low e r </w>    freq: 1
  low e s t </w>  freq: 1

  New pairs:
  (low, e) → 2   ← most frequent
```

**Iteration 3** — merge `(low, e)` → `lowe`:

```
  low </w>         freq: 3
  lowe r </w>      freq: 1
  lowe s t </w>    freq: 1
```

After N iterations the vocabulary contains:
```
  Characters  :  l, o, w, e, r, s, t
  Subwords    :  lo, low, lowe, lower, lowest
  Full words  :  low
```

#### BPE Merge Process Diagram

```
                     TRAINING CORPUS
                          │
                          ▼
              ┌─────────────────────┐
              │  Character-level    │
              │  vocabulary init    │
              │  + frequency count  │
              └──────────┬──────────┘
                         │
                         ▼
              ┌─────────────────────┐
              │  Find most frequent │◄──────┐
              │  adjacent pair      │       │
              └──────────┬──────────┘       │
                         │                  │
                         ▼                  │
              ┌─────────────────────┐       │
              │  Merge pair into    │       │ Repeat
              │  single new symbol  │       │ until vocab
              └──────────┬──────────┘       │ size reached
                         │                  │
                         ▼                  │
              ┌─────────────────────┐       │
              │  Record merge rule  │───────┘
              │  Add to vocab       │
              └──────────┬──────────┘
                         │
                         ▼
                  FINAL VOCABULARY
              (characters + subwords + words)
```

#### BPE at Inference (Encoding a New Word)

```
New word: "lowest"

Apply learned merge rules in order:
  Step 1: l o w e s t  →  lo w e s t    (rule: l+o → lo)
  Step 2: lo w e s t   →  low e s t     (rule: lo+w → low)
  Step 3: low e s t    →  lowe s t      (rule: low+e → lowe)
  Step 4: lowe s t     →  lowest        (rule: lowe+st → lowest)

Token IDs assigned to each piece → fed to embedding layer
```

### BPE Pros and Cons

```
  Pros
  ────────────────────────────────────────────
  + Handles unseen words via subword fallback
  + Vocabulary size is fixed and controllable
  + Balances word-level and char-level trade-off
  + Used by GPT-2, GPT-3, GPT-4, RoBERTa

  Cons
  ────────────────────────────────────────────
  - Requires pre-tokenization (whitespace split first)
  - Language-specific — not ideal for scripts without spaces
  - Merge order is greedy, not globally optimal
```

---

## 4. SentencePiece

SentencePiece (Kudo & Richardson, 2018) takes a different approach: it treats the
**entire raw text as a stream of Unicode characters** with no pre-tokenization step.
This makes it language-agnostic by design.

### The Key Difference from BPE

```
BPE Pipeline:
  Raw Text → Whitespace Split → Characters → BPE Merges → Tokens

SentencePiece Pipeline:
  Raw Text → Unicode Stream → SentencePiece Merges → Tokens
              (no whitespace split required)
```

Because there is no whitespace split, SentencePiece must encode word boundaries
itself. It does this using the `▁` (U+2581) prefix character.

### The ▁ Boundary Marker

```
Input sentence: "deep learning models"

SentencePiece output:
  ▁deep  ▁learn  ing  ▁model  s

  ▁ at the start  → this piece begins a new word
  No ▁            → this piece continues the previous word

Reconstruction:
  ▁deep           →  "deep"
  ▁learn + ing    →  "learning"
  ▁model + s      →  "models"
```

This design means the tokenizer can reconstruct the original text perfectly
from tokens alone, without needing to know where spaces were.

### SentencePiece Training Diagram

```
                     RAW TEXT CORPUS
               (no preprocessing, no splits)
                          │
                          ▼
              ┌─────────────────────────┐
              │  Unicode character      │
              │  stream + ▁ inserted    │
              │  before each word start │
              └──────────┬──────────────┘
                         │
                         ▼
              ┌─────────────────────────┐
              │  BPE or Unigram LM      │
              │  algorithm applied on   │◄──────┐
              │  the character stream   │       │ Repeat
              └──────────┬──────────────┘       │
                         │                      │
                         ▼                      │
              ┌─────────────────────────┐       │
              │  Merge / prune pieces   │───────┘
              │  based on frequency     │
              └──────────┬──────────────┘
                         │
                         ▼
                  FINAL VOCABULARY
          (▁word, subword, ▁sub, word pieces)
```

### SentencePiece Pros and Cons

```
  Pros
  ────────────────────────────────────────────
  + No language-specific pre-tokenization needed
  + Works natively on Chinese, Japanese, Hindi, Arabic etc.
  + ▁ marker enables perfect text reconstruction
  + Used by T5, XLM-RoBERTa, ALBERT, LLaMA, Gemma, Mistral

  Cons
  ────────────────────────────────────────────
  - Slightly slower to train than BPE
  - ▁ prefix can be confusing when inspecting tokens
  - Still not perfect on low-resource languages with small corpora
```

---

## 5. BPE vs SentencePiece — Comparison

```
┌─────────────────────┬──────────────────────┬────────────────────────┐
│ Aspect              │ BPE (tiktoken)        │ SentencePiece          │
├─────────────────────┼──────────────────────┼────────────────────────┤
│ Pre-tokenization    │ Whitespace split      │ Raw stream, none       │
│ Word boundary       │ Leading space " word" │ ▁ prefix on new word   │
│ Algorithm           │ Frequency merge       │ BPE or Unigram LM      │
│ Multilingual        │ Moderate              │ Excellent              │
│ Vocab size (typical)│ ~50K – 100K           │ ~32K – 250K            │
│ Used by             │ GPT-2/3/4, RoBERTa   │ T5, LLaMA, Gemma,      │
│                     │                      │ Mistral, XLM-RoBERTa   │
│ Reconstruction      │ Strip leading space   │ Replace ▁ with space   │
│ OOV handling        │ Subword fallback      │ Subword fallback        │
└─────────────────────┴──────────────────────┴────────────────────────┘
```

---

## 6. Same Sentence — Two Tokenizers

### English: "Unbelievably, deep learning"

```
  BPE (tiktoken)
  ──────────────────────────────────────────────────
  Input  : Unbelievably, deep learning
  Tokens : ['Un', 'belie', 'vably', ',', ' deep', ' learning']
  Count  : 6 tokens

  SentencePiece (xlm-roberta)
  ──────────────────────────────────────────────────
  Input  : Unbelievably, deep learning
  Tokens : ['▁Un', 'beli', 'eva', 'bly', ',', '▁deep', '▁learning']
  Count  : 7 tokens

  Observation:
  Same word "Unbelievably" → different split boundaries
  BPE:  Un | belie | vably
  SP:   Un | beli  | eva | bly
  Different tokens → different embeddings → different model behaviour
```

### Hindi: "नमस्ते दुनिया" (Hello World)

```
  BPE (tiktoken)
  ──────────────────────────────────────────────────
  Tokens : ['न', 'म', 'स', '्', '◌', 'े', ' ◌', '◌', 'ु', 'न', 'ि◌', '◌', 'ा']
  Count  : 13 tokens   ← fragmented byte-level pieces

  SentencePiece (xlm-roberta)
  ──────────────────────────────────────────────────
  Tokens : ['▁नम', 'स्ते', '▁दुनिया']
  Count  : 3 tokens   ← meaningful syllable-level pieces

  Observation:
  tiktoken was not trained heavily on Hindi → falls back to bytes
  SentencePiece was trained on 100 languages → clean subword split
  Fewer tokens = more efficient = better language understanding
```

---

## 7. Why Tokenization Matters for Embeddings

Every token ID gets looked up in the **embedding table** — a matrix of shape
`[vocab_size × embedding_dim]`. This is the first layer of every transformer.

```
  Token ID  →  Embedding Table Lookup  →  Dense Vector

  "king"  →  ID: 11734  →  [0.505,  0.686, -0.595, ...]  (50 or 768 dims)
  "Un"    →  ID:  1844  →  [0.023, -0.112,  0.341, ...]
  "belie" →  ID: 32898  →  [0.187,  0.045, -0.203, ...]
```

The model has **no concept of the original word** — it only sees token embeddings.
This is why tokenization quality directly affects model performance:

```
  Bad tokenization  →  noisy fragmented embeddings  →  harder to learn
  Good tokenization →  clean meaningful pieces      →  easier to learn
```

---

*References*
- Sennrich et al. (2016) — Neural Machine Translation of Rare Words with Subword Units (BPE)
- Kudo & Richardson (2018) — SentencePiece: A simple and language independent subword tokenizer
- tiktoken — https://github.com/openai/tiktoken
- SentencePiece — https://github.com/google/sentencepiece