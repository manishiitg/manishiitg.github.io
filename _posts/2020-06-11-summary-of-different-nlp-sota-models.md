# Short Summary of Different NLP Models 

check leader boards for different models 

https://super.gluebenchmark.com/leaderboard
https://gluebenchmark.com/leaderboard
https://rajpurkar.github.io/SQuAD-explorer/
https://paperswithcode.com/sota/document-summarization-on-cnn-daily-mail

## BERT

*Release date Nov 2, 2018 by Google AI*

Based on transformer

This is a Masked Language Model i.e MLM.

The first model realeased by google to cause the NLP revolution. 

Its trained on MLM i.e masking a word randomly in a sentence and predirecting it. Its reads sentence both front/back for prediction.

Trained on NSP i.e next sentence prediction as well. i.e given two sentences it will predict if the next sentence should follow the first sentence.

Can be used with a wide range of tasks NLP tasks. 

Some useful links to refer

https://ai.googleblog.com/2018/11/open-sourcing-bert-state-of-art-pre.html

http://jalammar.github.io/illustrated-bert/

https://mlexplained.com/2019/01/07/paper-dissected-bert-pre-training-of-deep-bidirectional-transformers-for-language-understanding-explained/

## Transformer XL

*Release date Jan 29, 2019 by Google AI*

This basically tries to solve the problem on longer sequence.
Model's like BERT are trained for shorter sequence lengths.
For longer sequences BERT divides the sentence into multiple segments. This has problems because the context in sentences get lost.

This the problem Transformer XL tries to solve.

Read more about this here 

https://ai.googleblog.com/2019/01/transformer-xl-unleashing-potential-of.html

https://medium.com/@mromerocalvo/dissecting-transformer-xl-90963e274bd7

https://mlexplained.com/2019/07/04/building-the-transformer-xl-from-scratch/

## XLNet

*Release date July 16, 2019*

This is based on top of Transformer XL and use Permutational Langauge Modelling. 

This is a very good read to understand XLNet https://mlexplained.com/2019/06/30/paper-dissected-xlnet-generalized-autoregressive-pretraining-for-language-understanding-explained/

XLNet achieves state-of-the-art performance (beats BERT) across 18 tasks https://github.com/zihangdai/xlnet

*Use sentpiece to tokenize data*


## Roberta

*Released 29 July 2019 by Facebook AI*

This is based on BERT but changes include training on much more data and for a much longer time. Also this trains only for MLM and not NSP like bert. 

> After implementing these design changes, our model delivered state-of-the-art performance on the MNLI, QNLI, RTE, STS-B, and RACE tasks and a sizable performance improvement on the GLUE benchmark. With a score of 88.5, RoBERTa reached the top position on the GLUE leaderboard, matching the performance of the previous leader, XLNet-Large. 

https://ai.facebook.com/blog/roberta-an-optimized-method-for-pretraining-self-supervised-nlp-systems/


## ALBERT - Light Bert

*Release date Dec 20, 2019 by Google AI*

This is simply a lighter version of BERT. 

> This is achieved by factorization of the embedding parametrization — the embedding matrix is split between input-level embeddings with a relatively-low dimension (e.g., 128), while the hidden-layer embeddings use higher dimensionalities (768 as in the BERT case, or more). With this step alone, ALBERT achieves an 80% reduction in the parameters of the projection block, at the expense of only a minor drop in performance — 80.3 SQuAD2.0 score, down from 80.4; or 67.9 on RACE, down from 68.2 — with all other conditions the same as for BERT.

https://ai.googleblog.com/2019/12/albert-lite-bert-for-self-supervised.html

There are many tasks like SQUAD, GLUE where albert scored ahead BERT. 


## DistilBert

This is basically a smaller version of BERT by huggingface. 
https://medium.com/huggingface/distilbert-8cf3380435b5

It the same model as BERT, using compression. 

> DistilBERT reduces the size of a BERT model by 40%, while retaining 97% of its language understanding capabilities and being 60% faster.



## GPT2 

For text generation. 
Need to understand how it works.
TODO:


## Reformer

*Release 16 Jan 2019, Google AI*

This is a new model based on Transformer for very long sequences upto a million words.
This can process entire books and it can be extended to audio/image sequences as well which are very long.

https://ai.googleblog.com/2020/01/reformer-efficient-transformer.html

This is a new model which is explored and can process very large sequences. It's not applied to any NLP tasks as such to get SOTA results. It's just a new model to process very long sequences. 


## T5: Text-to-Text Transformer

*Release Date 24 Feb 2020 Google AI*

> With T5, we propose reframing all NLP tasks into a unified text-to-text-format where the input and output are always text strings, in contrast to BERT-style models that can only output either a class label or a span of the input. Our text-to-text framework allows us to use the same model, loss function, and hyperparameters on any NLP task, including machine translation, document summarization, question answering, and classification tasks (e.g., sentiment analysis). We can even apply T5 to regression tasks by training it to predict the string representation of a number instead of the number itself.

>T5 is an extremely large new neural network model that is trained on a mixture of unlabeled text (the authors’ huge new C4 collection of English web text) and labeled data from popular natural language processing tasks, then fine-tuned individually for each of the tasks that they authors aim to solve.

T5 has given SoTA results across almost all nlp tasks

https://ai.googleblog.com/2020/02/exploring-transfer-learning-with-t5.html


## Electra

*Release date 10th March 2020 by Google AI*

In this model the main change is to change the way pretraing in done instead of MLM. A new way to pretain has been introduced which reduces the time and compute required to pretrain but still matches performance for models like xlnet

>ELECTRA — Efficiently Learning an Encoder that Classifies Token Replacements Accurately — is a novel pre-training method that outperforms existing techniques given the same compute budget. For example, ELECTRA matches the performance of RoBERTa and XLNet on the GLUE natural language understanding benchmark when using less than ¼ of their compute and achieves state-of-the-art results on the SQuAD question answering benchmark. ELECTRA’s excellent efficiency means it works well even at small scale — it can be trained in a few days on a single GPU to better accuracy than GPT, a model that uses over 30x more compute. 


## Longformer

*Release date 10th April 2020 by AllenAI*

This is able to process long documents with a sliding attention window. This model is mainly to process longer documents upto length of 4096.


https://github.com/allenai/longformer

https://mc.ai/longformer%e2%80%8a-%e2%80%8awhat-bert-should-have-been/




---

Any article giving summary of all models https://huggingface.co/transformers/summary.html by huggingface