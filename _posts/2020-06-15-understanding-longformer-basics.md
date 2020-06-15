# Understanding Longformer

This is the offical release of Longformer
[https://github.com/allenai/longformer](https://github.com/allenai/longformer)

Longformer solves the problem of long sequence using "sliding window" attention. 

Transformers like BERT are limited to a maximum of 512 token and even at 512 token the consume a lot of GPU.

Longormer solves this problem using "sliding attention window"


> Sliding Window Given the importance of local
context (Kovaleva et al., 2019), our attention pattern employs a fixed-size window attention surrounding each token. Using multiple stacked layers of such windowed attention results in a large
receptive field, where top layers have access to
all input locations and have the capacity to build
representations that incorporate information across
the entire input.


>Dilated Sliding Window To further increase the
receptive field without increasing computation, the
sliding window can be “dilated”. This is analogues
to dilated CNNs (van den Oord et al., 2016) where
the window has gaps of size dilation d (Fig. 2c).
Assuming a fixed d and w for all layers, the receptive field is ` × d × w, which can reach tens of
thousands of tokens even for small values of d.


> Global Attention In state-of-the-art BERT-style
models for natural language tasks, the optimal input representation differs from language modeling
and varies by task. For masked language modeling
(MLM), the model uses local context to predict the
masked word, while for classification, the model aggregates the representation of the whole sequence
into a special token ([CLS] in case of BERT). For
QA, the question and document are concatenated,
allowing the model to compare the question with
the document through self-attention.

[This blog](https://mc.ai/longformer%e2%80%8a-%e2%80%8awhat-bert-should-have-been/) does a very good explaining it.