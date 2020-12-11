## Summary Of Tasks任务总览

### 序列分类
序列分类旨在根据给定数量的类别对序列进行分类。比如GLUE.

下述是在sst-2进行微调的模型，识别序列情感类型是正还是负。
```
from transformers import pipeline

nlp = pipeline("sentiment-analysis")

result = nlp("I hate you")[0]
print(f"label: {result['label']}, with score: {round(result['score'], 4)}")

===> label: NEGATIVE, with score: 0.9991

result = nlp("I love you")[0]
print(f"label: {result['label']}, with score: {round(result['score'], 4)}")

===> label: POSITIVE, with score: 0.9999
```

下述是判断两个句子是不是互相解释的情形。其过程大致如下：
- 首先从checkpoint中实例化分词器和模型。
- 使用特定于模型的分隔符、Token类型Id和注意力Mask，将两个句子构建成一个序列。
- 将序列传递给模型，将其分类为两个类别之一：0（不是）或1（是）
- 计算结果的softmax以获得类的概率
- 打印结果
```
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

tokenizer = AutoTokenizer.from_pretrained("bert-base-cased-finetuned-mrpc")
model = AutoModelForSequenceClassification.from_pretrained("bert-base-cased-finetuned-mrpc")

classes = ["not paraphrase", "is paraphrase"]

sequence_0 = "The company HuggingFace is based in New York City"
sequence_1 = "Apples are especially bad for your health"
sequence_2 = "HuggingFace's headquarters are situated in Manhattan"

paraphrase = tokenizer(sequence_0, sequence_2, return_tensors="pt")
not_paraphrase = tokenizer(sequence_0, sequence_1, return_tensors="pt")

paraphrase_classification_logits = model(**paraphrase).logits
not_paraphrase_classification_logits = model(**not_paraphrase).logits

paraphrase_results = torch.softmax(paraphrase_classification_logits, dim=1).tolist()[0]
not_paraphrase_results = torch.softmax(not_paraphrase_classification_logits, dim=1).tolist()[0]

# Should be paraphrase
for i in range(len(classes)):
     print(f"{classes[i]}: {int(round(paraphrase_results[i] * 100))}%")

===> not paraphrase: 10%
===> is paraphrase: 90%

# Should not be paraphrase
for i in range(len(classes)):
    print(f"{classes[i]}: {int(round(not_paraphrase_results[i] * 100))}%")
===> not paraphrase: 94%
===> is paraphrase: 6%
```

### 提取式QA
提取式QA是从给定问题的文本中提取答案的任务。比如SQuAD。

(注：下面这段代码我运行不起来)
下面是使用pipeline进行QA的示例，它使用了从SQuAD上微调的模型。
```
from transformers import pipeline

nlp = pipeline("question-answering")

context = r"""
    Extractive Question Answering is the task of extracting an answer from a text given a question. An example of a
    question answering dataset is the SQuAD dataset, which is entirely based on that task. If you would like to fine-tune
    a model on a SQuAD task, you may leverage the examples/question-answering/run_squad.py script.
    """
```
返回提取的答案，置信度，在文中的开始和结束位置
```
result = nlp(question="What is extractive question answering?", context=context)
print(f"Answer: '{result['answer']}', score: {round(result['score'], 4)}, start: {result['start']}, end: {result['end']}")

===> Answer: 'the task of extracting an answer from a text given a question.', score: 0.6226, start: 34, end: 96

result = nlp(question="What is a good example of a question answering dataset?", context=context)
print(f"Answer: '{result['answer']}', score: {round(result['score'], 4)}, start: {result['start']}, end: {result['end']}")

===> Answer: 'SQuAD dataset,', score: 0.5053, start: 147, end: 161
```
下面是使用模型进行QA的过程，大致如下：
- 首先从checkpoint中实例化分词器和模型。
- 定义文本和一些问题
- 遍历问题，使用特定于模型的分隔符、Token类型Id和注意力Mask，将两个句子构建成一个序列。
- 将序列传递给模型，模型的输出是整个回答的起点和终点范围
- 计算结果的softmax以获得在token上的概率
- 从起始和终点获取token，将他们转化为字符串
- 打印结果
```
from transformers import AutoTokenizer, AutoModelForQuestionAnswering
import torch

tokenizer = AutoTokenizer.from_pretrained("bert-large-uncased-whole-word-masking-finetuned-squad")
model = AutoModelForQuestionAnswering.from_pretrained("bert-large-uncased-whole-word-masking-finetuned-squad")

text = r"""
    🤗 Transformers (formerly known as pytorch-transformers and pytorch-pretrained-bert) provides general-purpose
    architectures (BERT, GPT-2, RoBERTa, XLM, DistilBert, XLNet…) for Natural Language Understanding (NLU) and Natural
    Language Generation (NLG) with over 32+ pretrained models in 100+ languages and deep interoperability between
    TensorFlow 2.0 and PyTorch.
"""

questions = [
    "How many pretrained models are available in 🤗 Transformers?",
    "What does 🤗 Transformers provide?",
    "🤗 Transformers provides interoperability between which frameworks?",
]

for question in questions:
    inputs = tokenizer(question, text, add_special_tokens=True, return_tensors="pt")
    input_ids = inputs["input_ids"].tolist()[0]

    text_tokens = tokenizer.convert_ids_to_tokens(input_ids)
    outputs = model(**inputs)
    answer_start_scores = outputs.start_logits
    answer_end_scores = outputs.end_logits

    answer_start = torch.argmax(
        answer_start_scores
    )  # Get the most likely beginning of answer with the argmax of the score
    answer_end = torch.argmax(answer_end_scores) + 1  # Get the most likely end of answer with the argmax of the score

    answer = tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(input_ids[answer_start:answer_end]))

    print(f"Question: {question}")
    print(f"Answer: {answer}")

===> Question: How many pretrained models are available in 🤗 Transformers?
===> Answer: over 32 +
===> Question: What does 🤗 Transformers provide?
===> Answer: general - purpose architectures
===> Question: 🤗 Transformers provides interoperability between which frameworks?
===> Answer: tensorflow 2 . 0 and pytorch
```
注：上面的代码在获取answer_start_scores和answer_end_scores的时候会报错，因为说outputs是一个tuple。我自己改成了outputs[0]和outputs[1]就可以运行了。