# Intro to SQUAD

### Basics
SQUAD is a question answering dataset (QA) 

The official dataset can be found here https://rajpurkar.github.io/SQuAD-explorer/

The dataset is based on crowdsources answers for wikipedia articles on specific question.
So basically this means, the dataset has 
- title  => title of the article
- paragaph => article description this usually a single paragraph from the wiki article
- question => a specific question
- answer => this contains answer for that question and more specifically the start index of the answer and the actual answer string


SQUAD v2.0 dataset has additional question which are impossible to answer. SQUAD v1.0 has questions only which can be answed, but v2.0 has additional question which cannot be answered at all.

The dataset doesn't have multipe choice questions. 

You can view the dataset here easiy https://huggingface.co/nlp/viewer/

### Colab Notebook - Training

To train BERT or (other huggingface models) on SQUAD v2.0 dataset. 

https://colab.research.google.com/drive/1kuyq_I41cKiTeXCC2OxOK1qkmcgR0MJb?usp=sharing

This code is based on official example from huggingface. 
You can run and understand this code to get more indepth of how SQUAD works.

Also go through this to understand more deeply how the data is converted and fed into models https://github.com/huggingface/transformers/blob/f9414f7553d3f1872b372990ef03205c0d1141df/src/transformers/data/processors/squad.py


### Predictions using - Pretained 
In the above colab i have done predections using different ways but to demonstrate the most simplest way

```python
from transformers import pipeline
nlp = pipeline('question-answering' , model='bert-large-cased-whole-word-masking-finetuned-squad', tokenizer='bert-large-cased-whole-word-masking-finetuned-squad')

nlp({
    'question': 'how much work experiance?',
    'context': 'I have been working with excellence for 10 years now and before that i worked for 2years with headstrong ltd'
})
```



```
{'answer': '10 years', 'end': 61, 'score': 0.5556458644379347, 'start': 53}
```

```python
nlp({
    'question': 'how much work experiance do i have working at headstrong',
    'context': 'I have been working with excellence for 10 years now and before that i worked for 2years with headstrong ltd'
})
```

```
{'answer': '2years', 'end': 101, 'score': 0.9086288980343227, 'start': 95}
```

This words well out of the box!


