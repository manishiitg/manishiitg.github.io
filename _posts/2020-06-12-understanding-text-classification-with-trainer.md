# Notes Text Classification understanding on huggingface features/examples with Trainer


I am just taking some notes here as i understand in-depth 
how examples/features/datasets are working.
How the tensors are actually passed onto the Transformer models like BERT

## Text Classification

Full Colab [https://colab.research.google.com/drive/1IC5Be6Y_ZAh_z51n8RESUuKfCO4Hg8iv?usp=sharing](https://colab.research.google.com/drive/1IC5Be6Y_ZAh_z51n8RESUuKfCO4Hg8iv?usp=sharing)


#### Step 1
First we need to convert our data which can be any format json, csv, tsv etc to "Examples"

For this we extend the "DataProcessor" class. 
Purpose of "DataProcessor" is to mainly generate "train' , "dev' , 'test' examples from a dataset.

```python

from transformers import DataProcessor, InputExample, InputFeatures, PreTrainedTokenizer, glue_convert_examples_to_features
from torch.utils.data.dataset import Dataset
from typing import List, Optional, Union
from enum import Enum
from filelock import FileLock
import time
import torch


class Split(Enum):
    train = "train"
    dev = "dev"
    test = "test"

class CustomProcessor(DataProcessor):

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "test.tsv")), "test")

    def get_labels(self):
        """See base class."""
        # for read in from your train files or seperate json file for labels
        return ["label1" , "label2"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training, dev and test sets."""
        examples = []
        text_index = 1 if set_type == "test" else 0
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, i)
            text_a = line[text_index]
            label = None if set_type == "test" else line[1]
            examples.append(InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
        return examples

```

Here we are using on class called "InputExample"
This class just stores the values nothing else.

```python
@dataclass
class InputExample:
    """
    A single training/test example for simple sequence classification.
    Args:
        guid: Unique id for the example.
        text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
        text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
        label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
    """

    guid: str
    text_a: str
    text_b: Optional[str] = None
    label: Optional[str] = None

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(dataclasses.asdict(self), indent=2) + "\n"
```

## Step2

Next we need to convert the examples into features.

This is done using the inbuilt function "glue_convert_examples_to_features" [https://github.com/huggingface/transformers/blob/6449c494d0f40f0b70442f3da9e61f042ff807a8/src/transformers/data/processors/glue.py#L107](https://github.com/huggingface/transformers/blob/6449c494d0f40f0b70442f3da9e61f042ff807a8/src/transformers/data/processors/glue.py#L107)


The input parameter to this are

```python

examples: List[InputExample],
    tokenizer: PreTrainedTokenizer,
    max_length: Optional[int] = None,
    task=None,
    label_list=None,
    output_mode=None,

```

here tokenizer will be the model dependent tokenizer having the base class [https://github.com/huggingface/transformers/blob/9f5d5a531d769d07403f59661884e254f8420afe/src/transformers/tokenization_utils.py#L2224](https://github.com/huggingface/transformers/blob/9f5d5a531d769d07403f59661884e254f8420afe/src/transformers/tokenization_utils.py#L2224
)


This function is important as it tokenizes your data and returns a list of 'InputFeatures'

[https://github.com/huggingface/transformers/blob/6449c494d0f40f0b70442f3da9e61f042ff807a8/src/transformers/data/processors/utils.py#L56](https://github.com/huggingface/transformers/blob/6449c494d0f40f0b70442f3da9e61f042ff807a8/src/transformers/data/processors/utils.py#L56)

At this stage, we only have integers, we don't have tensors yet.

```python

"""
    A single set of features of data.
    Property names are the same names as the corresponding inputs to a model.
    Args:
        input_ids: Indices of input sequence tokens in the vocabulary.
        attention_mask: Mask to avoid performing attention on padding token indices.
            Mask values selected in ``[0, 1]``:
            Usually  ``1`` for tokens that are NOT MASKED, ``0`` for MASKED (padded) tokens.
        token_type_ids: (Optional) Segment token indices to indicate first and second
            portions of the inputs. Only some models use them.
        label: (Optional) Label corresponding to the input. Int for classification problems,
            float for regression problems.
    """

    input_ids: List[int]
    attention_mask: Optional[List[int]] = None
    token_type_ids: Optional[List[int]] = None
    label: Optional[Union[int, float]] = None

```

### Step3 

Next these features are directly fed into Trainer.
Also with model and tokenizer. 

[https://github.com/huggingface/transformers/blob/d541938c48f759522f81fa177aae49098e0e0149/src/transformers/trainer.py#L153](https://github.com/huggingface/transformers/blob/d541938c48f759522f81fa177aae49098e0e0149/src/transformers/trainer.py#L153)


Trainer has been recently released. This classes takes our dataset (i.e list of features), model, tokenizer and does the training. 
It handles all the stuff for gpu, tpu, saving checkpoints etc etc for you.


Next the main thing i am look at is how the "features" i.e list of integers is converted to "tensors" and passed to a model.

This is where the "DataCollator" class comes into picture.
[https://github.com/huggingface/transformers/blob/d541938c48f759522f81fa177aae49098e0e0149/src/transformers/data/data_collator.py#L32](https://github.com/huggingface/transformers/blob/d541938c48f759522f81fa177aae49098e0e0149/src/transformers/data/data_collator.py#L32)

We can also define our custom collator if need. 

```python

for k, v in vars(first).items():
            if k not in ("label", "label_ids") and v is not None and not isinstance(v, str):
                batch[k] = torch.tensor([getattr(f, k) for f in features], dtype=torch.long)
        return batch

```


## Step4 

```python

model.train()
for k, v in inputs.items():
    inputs[k] = v.to(self.args.device)

outputs = model(**inputs)

```

also the model "BertForSequenceClassification" takes input as 

```python

def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
    ):

```


This is the flow of data upto the training step










