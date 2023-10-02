# score-based-diffusion-lm
A score-based Diffusion LM approach.

## What are the aims of this repository?

Using [simple-diffusion-lm](https://github.com/thorinf/simple-diffusion-lm), this repository attempts a score-based
approach.
There are some additional design changes, the biggest change is to a multi-level diffusion step - more details below.

## How to Run:

### Environment Setup:

Aside from PyTorch, the training script requires `tqdm` and two other packages;
[SentencePiece](https://github.com/google/sentencepiece), and
[RotaryEmbedding](https://github.com/lucidrains/rotary-embedding-torch).
They can be installed with the following commands:

```commandline
pip install sentencepiece
pip install rotary-embedding-torch
pip install tqdm
```

### Pre-Processing:

First generate a `.txt` corpus where each line is an example.
It's recommended to apply some normalisation on the text so the data is quite clean for the next step and training, e.g.
lower-case, change numbers to words, removing unnecessary symbols.
The training script won't perform these normalisations, so data should be cleaned externally.

With a clean text corpus, the SentencePiece model can then be trained.
Follow the guides on [their repository](https://github.com/google/sentencepiece)
or [here on PyPI](https://pypi.org/project/sentencepiece/).
If the text corpus is very large, then creating a subset of the text can get around memory issues.
Here is an exert from the script that created the BPE model: 

```python
spm.SentencePieceTrainer.train(
    input=text_path,
    model_prefix=name,
    model_type='bpe',
    vocab_size=size,
    user_defined_symbols=[str(i) for i in range(10)],
    bos_id=0,
    eos_id=1,
    pad_id=2,
    unk_id=3
)
```

### Training:

The model can be trained with the command:

```commandline
python train.py -d=TXT_CORPUS -spm=SPM_MODEL -mdir=MODEL_DIRECTORY
```

There's a bunch of other arguments which can be altered, but above is enough to get the model working.

## Design & Implementation Notes:

Many of the details in architecture are mirrors of [simple-diffusion-lm](https://github.com/thorinf/simple-diffusion-lm),
so here only the changes differences will be listed.

### Sequence Packing:
When training on variable length sequences there can often be redundant computation due to padding of short sequences.
To combat this, if sequences are sufficiently short they are concatenated together. 
These concatenated sequences will not be longer than the maximum sequence length for that batch - 
this max length is capped during training.
The attention mask is created in the dataloader such that these concatenated sequences do not affect one another.

This increases the number of trainable tokens in each batch:

| Batch Size | Max Length | Packing | Tokens | Efficiency |
|------------|------------|--------|--------|------------|
| 128        | 64         | No     | 6820   | 83%        |
| 128        | 64         | Yes    | 7420   | 90%        |

### Multi-Level Diffusion:
One of the main inspirations for this approach is [AR-Diffusion][8]. 
The rate that words diffuse varies depends on their complexity, 
less informative words typically diffuse faster ([see here][8], [and here][9]).
In [AR-Diffusion][8], a multi-level diffusion strategy is used where each index in the sequence generation can
have its own diffusion time-step. 
This means that the velocity of diffusion can be made faster for tokens.
In their case the earlier in the sequence, the faster the diffusion, thus [AR-Diffusion][8].

#### Training at Random Diffusion Steps:
What if instead of [AR-Diffusion][8] there were instead other diffusion strategies that changes the velocity
in a way that's independent of sequence position?
For example, it may be good to accelerate the diffusion for positions that consistently predict the same token with
a high probability. 
Possible complex strategies would require the model to be robust to a variety of noise levels in the sequence.
To accommodate this possibility the training loss uses random amounts of scheduled noise for each embedding vector.
Each batch is using a 2D array of perturbation scaling, instead of the conventional 1D.

#### Conditioning:
In previous works, such as [CDCD][3], conditional embeddings are also fed into the model with a corresponding
conditional mask.
As this implementation targets a model that can use multi-level diffusion, the conditional masking is handled differently.
Any conditional positions have the diffusion step set to 0, i.e. no noise, during training.
The model isn't explicitly told that those positions are conditional, other than their diffusion has "completed".

### Noise Scheduling:
[Democratized Diffusion Language Model][10] suggests different noise levels to what was proposed in [CDCD][3]. 
This implementation adopts the same, lower noise levels. 


[1]: <https://arxiv.org/abs/2208.04202> "Analog Bits: Generating discrete data using diffusion models with self-conditioning"

[2]: <https://arxiv.org/abs/2212.09412> "Difformer: Empowering Diffusion Models on the Embedding Space for Text Generation"

[3]: <https://arxiv.org/abs/2211.15089> "Continuous diffusion for categorical data"

[4]: <https://arxiv.org/abs/2205.14217> "Diffusion-LM Improves Controllable Text Generation"

[5]: <https://arxiv.org/abs/2301.10972> "On the Importance of Noise Scheduling for Diffusion Models"

[6]: <https://arxiv.org/abs/2104.09864> "RoFormer: Enhanced Transformer with Rotary Position Embedding"

[7]: <https://arxiv.org/abs/1709.07871> "FiLM: Visual Reasoning with a General Conditioning Layer"

[8]: <https://arxiv.org/abs/2305.09515> "AR-Diffusion: Auto-Regressive Diffusion Model for Text Generation"

[9]: <https://arxiv.org/abs/2304.04746> "A Cheaper and Better Diffusion Language Model with Soft-Masked Noise"

[10]: <https://arxiv.org/abs/2305.10818> "Democratized Diffusion Language Model"