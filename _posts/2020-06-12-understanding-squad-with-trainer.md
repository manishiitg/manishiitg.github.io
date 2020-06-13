# Understanding how SQUAD works with trainer.

taking some notes here for my references 

https://colab.research.google.com/drive/1P-k91PvqRMdaoySs08GuzJfoz30cseA4?usp=sharing


### BertForQuestionAnswering

This is the model for QA for Bert.

```python

class BertForQuestionAnswering(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.bert = BertModel(config)
        self.qa_outputs = nn.Linear(config.hidden_size, config.num_labels)

        self.init_weights()


```

So basically BertForQA uses the base BertModel and then applies a Liner layer from the hidden_size.

Here num_labels = 2 always. and it corrsponse the start_index and end_index.

So basically the model takes input text and predirects the start_index and end_index of the answer.


The forward function of the model can be seen here https://github.com/huggingface/transformers/blob/d541938c48f759522f81fa177aae49098e0e0149/src/transformers/modeling_bert.py#L1578

This is what the model returns

```python

return outputs  # (loss), start_logits, end_logits, (hidden_states), (attentions)

```



### SquadV2Processor

Huggingface provides a processor that can read both SQUAD v1.0 and SQUAD v2.0 files 

https://github.com/huggingface/transformers/blob/f9414f7553d3f1872b372990ef03205c0d1141df/src/transformers/data/processors/squad.py#L552

This again creates a list of Examples 

```python

SquadExample(
    qas_id=qas_id,
    question_text=question_text,
    context_text=context_text,
    answer_text=answer_text,
    start_position_character=start_position_character,
    title=title,
    is_impossible=is_impossible,
    answers=answers,
)

```

Another thing to note is that, SquadExample internally creates doc_tokens. This is basically a list of all words in the input "context". Also based the on answer the start_position and end_position are also updated.
https://github.com/huggingface/transformers/blob/f9414f7553d3f1872b372990ef03205c0d1141df/src/transformers/data/processors/squad.py#L645





Next, the squadExamples are converted to features through the function  https://github.com/huggingface/transformers/blob/f9414f7553d3f1872b372990ef03205c0d1141df/src/transformers/data/processors/squad.py#L86

There is lot of code going here for different conditions but the most important is 

```python

encoded_dict = tokenizer.encode_plus(
            truncated_query if tokenizer.padding_side == "right" else span_doc_tokens,
            span_doc_tokens if tokenizer.padding_side == "right" else truncated_query,
            max_length=max_seq_length,
            return_overflowing_tokens=True,
            pad_to_max_length=True,
            stride=max_seq_length - doc_stride - len(truncated_query) - sequence_pair_added_tokens,
            truncation_strategy="only_second" if tokenizer.padding_side == "right" else "only_first",
            return_token_type_ids=True,
        )

```

In this the question_text and the context are encoded together.

To understand what these different options mean see the doc here https://github.com/huggingface/transformers/blob/9f5d5a531d769d07403f59661884e254f8420afe/src/transformers/tokenization_utils.py#L1496


Another thing to note here is that, SquadFeatures are just lists of numbers.

### SquadDataCollator

This is a custom class. This is to convert the features to tensors.

```python

@dataclass
class SquadDataCollator(DataCollator):
    def collate_batch(self, features: List) -> Dict[str, torch.Tensor]:
        # taken from https://github.com/huggingface/transformers/blob/5daca95dddf940139d749b1ca42c59ebc5191979/src/transformers/data/processors/squad.py#L325
        # and https://github.com/huggingface/transformers/blob/5daca95dddf940139d749b1ca42c59ebc5191979/src/transformers/data/processors/squad.py#L325
        # first = features[0]

        

        # for f in features:
        #   logger.warning("unique id collator %s", f.unique_id)
        #   logger.warning("example_index collator %s", f.example_index)

        unique_ids = torch.tensor([f.unique_id for f in features])

        all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
        all_attention_masks = torch.tensor([f.attention_mask for f in features], dtype=torch.long)
        all_token_type_ids = torch.tensor([f.token_type_ids for f in features], dtype=torch.long)
        all_cls_index = torch.tensor([f.cls_index for f in features], dtype=torch.long)
        all_p_mask = torch.tensor([f.p_mask for f in features], dtype=torch.float)

        

        # if hasattr(first, "start_position") and first.start_position is not None:
        all_start_positions = torch.tensor([f.start_position for f in features], dtype=torch.long)
        all_end_positions = torch.tensor([f.end_position for f in features], dtype=torch.long)
        all_example_index = torch.arange(all_input_ids.size(0), dtype=torch.long)

        return {
            "input_ids" : all_input_ids,
            "attention_mask" : all_attention_masks,
            "token_type_ids" : all_token_type_ids,
            "start_positions" : all_start_positions,
            "end_positions" : all_end_positions,
            "cls_index": all_cls_index,
            "p_mask": all_p_mask,
            "example_index" : all_example_index,
            "unique_ids" : unique_ids
        }

```

### Custom Trainer

Depending on the input to the model, there were some changes required. So i have overriden the training_step of the model.

```python

class SquadTrainer(Trainer):
    model_args: ModelArguments
    def __init__(self, model_args, **kwargs):
        self.model_args = model_args
        super().__init__(**kwargs)

    def _training_step(
        self, model: nn.Module, inputs: Dict[str, torch.Tensor], optimizer: torch.optim.Optimizer
    ) -> float:
        model.train()
        for k, v in inputs.items():
          inputs[k] = v.to(self.args.device)

        del inputs["unique_ids"]
        del inputs["example_index"]

        # this can be handled at dataset level as well. 
        # no need to extend SquadTrainer
        

        outputs = model(**inputs)
        loss = outputs[0]  # model outputs are always tuple in transformers (see doc)

        if self.args.n_gpu > 1:
            loss = loss.mean()  # mean() to average on multi-gpu parallel training
        if self.args.gradient_accumulation_steps > 1:
            loss = loss / self.args.gradient_accumulation_steps

        if self.args.fp16:
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()

        return loss.item()

```


# Evaluation 

The evaluation script works like this 

```python

outputs = model(**inputs)

unique_ids = unique_ids.cpu().numpy()
for i,unique_id in enumerate(unique_ids):
output = [to_list(output[i]) for output in outputs]

# Some models (XLNet, XLM) use 5 arguments for their predictions, while the other "simpler"
# models only use two.
if len(output) >= 5:
    start_logits = output[0]
    start_top_index = output[1]
    end_logits = output[2]
    end_top_index = output[3]
    cls_logits = output[4]

    result = SquadResult(
        unique_id,
        start_logits,
        end_logits,
        start_top_index=start_top_index,
        end_top_index=end_top_index,
        cls_logits=cls_logits,
    )

else:
    start_logits, end_logits = output
    result = SquadResult(unique_id, start_logits, end_logits)

all_results.append(result)

```

with this basically we create a list of SquadResult.
SquadResult has "unique_id' of the feature.
start_logits and end_logits which have been predicted by the model.








