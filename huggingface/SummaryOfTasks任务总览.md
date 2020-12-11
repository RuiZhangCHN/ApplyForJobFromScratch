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

### Masked语言模型
Mask语言模型是指通过随机的掩盖掉序列中的一部分token，让模型去填补这些token。这种语言模型允许模型attend到前向或者后向的上下文。

以下是使用pipeline执行填补的一个示例
```
from transformers import pipeline
from pprint import pprint

nlp = pipeline("fill-mask")
pprint(nlp(f"HuggingFace is creating a {nlp.tokenizer.mask_token} that the community uses to solve NLP tasks."))

===> [{'score': 0.1792745739221573,
===>   'sequence': '<s>HuggingFace is creating a tool that the community uses to '
===>               'solve NLP tasks.</s>',
===>   'token': 3944,
===>   'token_str': 'Ġtool'},
===>  {'score': 0.11349421739578247,
===>   'sequence': '<s>HuggingFace is creating a framework that the community uses '
===>               'to solve NLP tasks.</s>',
===>   'token': 7208,
===>   'token_str': 'Ġframework'},
===>  {'score': 0.05243554711341858,
===>   'sequence': '<s>HuggingFace is creating a library that the community uses to '
===>               'solve NLP tasks.</s>',
===>   'token': 5560,
===>   'token_str': 'Ġlibrary'},
===>  {'score': 0.03493533283472061,
===>   'sequence': '<s>HuggingFace is creating a database that the community uses '
===>               'to solve NLP tasks.</s>',
===>   'token': 8503,
===>   'token_str': 'Ġdatabase'},
===>  {'score': 0.02860250137746334,
===>   'sequence': '<s>HuggingFace is creating a prototype that the community uses '
===>               'to solve NLP tasks.</s>',
===>   'token': 17715,
===>   'token_str': 'Ġprototype'}]
```

以下是执行maskedLM建模的示例，该过程如下：
- 加载分词器和模型
- 使用带mask的token定义一个序列
- 将序列编码为ID列表，在该列表中找到被mask的token的位置
- 在mask token位置处进行检索检测：该张量和词汇表大小相同，其值是每个词语的分数。具有更高置信度的词语会有更高的分数。
- 使用topk检索前5个token
- 使用token替换mask并打印结果
```
from transformers import AutoModelWithLMHead, AutoTokenizer
import torch

tokenizer = AutoTokenizer.from_pretrained("distilbert-base-cased")
model = AutoModelWithLMHead.from_pretrained("distilbert-base-cased")

sequence = f"Distilled models are smaller than the models they mimic. Using them instead of the large versions would help {tokenizer.mask_token} our carbon footprint."

input = tokenizer.encode(sequence, return_tensors="pt")
mask_token_index = torch.where(input == tokenizer.mask_token_id)[1]

token_logits = model(input).logits
mask_token_logits = token_logits[0, mask_token_index, :]

top_5_tokens = torch.topk(mask_token_logits, 5, dim=1).indices[0].tolist()

for token in top_5_tokens:
    print(sequence.replace(tokenizer.mask_token, tokenizer.decode([token])))

===> Distilled models are smaller than the models they mimic. Using them instead of the large versions would help reduce our carbon footprint.
===> Distilled models are smaller than the models they mimic. Using them instead of the large versions would help increase our carbon footprint.
===> Distilled models are smaller than the models they mimic. Using them instead of the large versions would help decrease our carbon footprint.
===> Distilled models are smaller than the models they mimic. Using them instead of the large versions would help offset our carbon footprint.
===> Distilled models are smaller than the models they mimic. Using them instead of the large versions would help improve our carbon footprint.
```

### 因果语言模型
因果语言模型通常用在生成，根据前文预测下一个token。也就是说它只能够attend到左边已经产生的文本。通常，下一个标记是通过从模型根据输入序列生成的最后一个隐藏状态的对数中采样来预测的。

下面是采用topk_top_p_filtering()方法采样到下一个token的示例
```
from transformers import AutoModelWithLMHead, AutoTokenizer, top_k_top_p_filtering
import torch
from torch.nn import functional as F

tokenizer = AutoTokenizer.from_pretrained("gpt2")
model = AutoModelWithLMHead.from_pretrained("gpt2")

sequence = f"Hugging Face is based in DUMBO, New York City, and "

input_ids = tokenizer.encode(sequence, return_tensors="pt")

# get logits of last hidden state
next_token_logits = model(input_ids).logits[:, -1, :]

# filter
filtered_next_token_logits = top_k_top_p_filtering(next_token_logits, top_k=50, top_p=1.0)

# sample
probs = F.softmax(filtered_next_token_logits, dim=-1)
next_token = torch.multinomial(probs, num_samples=1)

generated = torch.cat([input_ids, next_token], dim=-1)

resulting_string = tokenizer.decode(generated.tolist()[0])

print(resulting_string)

===> Hugging Face is based in DUMBO, New York City, and has
```

### 文本生成
文本生成就是给定上下文产生下文的任务。下面是使用pipeline和GPT-2进行文本生成的示例
```
from transformers import pipeline

text_generator = pipeline("text-generation")
print(text_generator("As far as I am concerned, I will", max_length=50, do_sample=False))

===> [{'generated_text': 'As far as I am concerned, I will be the first to admit that I am not a fan of the idea of a "free market." I think that the idea of a free market is a bit of a stretch. I think that the idea'}]
```

下面是使用XLNet生成的示例，它读入了更大的上下文
```
from transformers import AutoModelWithLMHead, AutoTokenizer

model = AutoModelWithLMHead.from_pretrained("xlnet-base-cased")
tokenizer = AutoTokenizer.from_pretrained("xlnet-base-cased")

# Padding text helps XLNet with short prompts - proposed by Aman Rusia in https://github.com/rusiaaman/XLNet-gen#methodology
PADDING_TEXT = """In 1991, the remains of Russian Tsar Nicholas II and his family
    (except for Alexei and Maria) are discovered.
    The voice of Nicholas's young son, Tsarevich Alexei Nikolaevich, narrates the
    remainder of the story. 1883 Western Siberia,
    a young Grigori Rasputin is asked by his father and a group of men to perform magic.
    Rasputin has a vision and denounces one of the men as a horse thief. Although his
    father initially slaps him for making such an accusation, Rasputin watches as the
    man is chased outside and beaten. Twenty years later, Rasputin sees a vision of
    the Virgin Mary, prompting him to become a priest. Rasputin quickly becomes famous,
    with people, even a bishop, begging for his blessing. <eod> </s> <eos>"""

prompt = "Today the weather is really nice and I am planning on "
inputs = tokenizer.encode(PADDING_TEXT + prompt, add_special_tokens=False, return_tensors="pt")

prompt_length = len(tokenizer.decode(inputs[0], skip_special_tokens=True, clean_up_tokenization_spaces=True))
outputs = model.generate(inputs, max_length=250, do_sample=True, top_p=0.95, top_k=60)
generated = prompt + tokenizer.decode(outputs[0])[prompt_length:]

print(generated)

===> Today the weather is really nice and I am planning on anning on taking a nice...... of a great time!<eop>...............
```

### 命名实体识别
就是从句子中识别人名地名机构名的任务。例如CoNLL-2003数据集。

以下是使用pipeline实现NER的一个示例，一共把序列分成9种类别：
- O, 非实体
- B-MIS, 另一个混合实体的起始
- I-MIS, 混合实体
- B-PER, 另一个人名的起始
- I-PER, 人名
- B-ORG, 另一个机构名的起始
- I-ORG, 机构名
- B-LOC, 另一个地点的起始
- I-LOC, 地点
```
from transformers import pipeline

nlp = pipeline("ner")
sequence = "Hugging Face Inc. is a company based in New York City. Its headquarters are in DUMBO, therefore very"
           "close to the Manhattan Bridge which is visible from the window."

print(nlp(sequence))

===> [
===>     {'word': 'Hu', 'score': 0.9995632767677307, 'entity': 'I-ORG'},
===>     {'word': '##gging', 'score': 0.9915938973426819, 'entity': 'I-ORG'},
===>     {'word': 'Face', 'score': 0.9982671737670898, 'entity': 'I-ORG'},
===>     {'word': 'Inc', 'score': 0.9994403719902039, 'entity': 'I-ORG'},
===>     {'word': 'New', 'score': 0.9994346499443054, 'entity': 'I-LOC'},
===>     {'word': 'York', 'score': 0.9993270635604858, 'entity': 'I-LOC'},
===>     {'word': 'City', 'score': 0.9993864893913269, 'entity': 'I-LOC'},
===>     {'word': 'D', 'score': 0.9825621843338013, 'entity': 'I-LOC'},
===>     {'word': '##UM', 'score': 0.936983048915863, 'entity': 'I-LOC'},
===>     {'word': '##BO', 'score': 0.8987102508544922, 'entity': 'I-LOC'},
===>     {'word': 'Manhattan', 'score': 0.9758241176605225, 'entity': 'I-LOC'},
===>     {'word': 'Bridge', 'score': 0.990249514579773, 'entity': 'I-LOC'}
===> ]
```

基本流程
- 加载分词器和模型
- 定义用于训练模型的标签列表
- 将单词拆分为标记，以便可以将其映射到预测。首先，我们会对整个序列进行完全编码和解码，这是一个小技巧，因此我们只剩下一个包含特殊标记的字符串。将序列编码为ID（会自动添加特殊标记）。
- 通过将输入传递给模型并获取第一个输出来检索预测。这样就为每个令牌分配了9种可能的类别。我们使用argmax为每个令牌检索最可能的类。
- 将每个令牌及其预测一起压缩并打印。
```
from transformers import AutoModelForTokenClassification, AutoTokenizer
import torch

model = AutoModelForTokenClassification.from_pretrained("dbmdz/bert-large-cased-finetuned-conll03-english")
tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")

label_list = [
    "O",       # Outside of a named entity
    "B-MISC",  # Beginning of a miscellaneous entity right after another miscellaneous entity
    "I-MISC",  # Miscellaneous entity
    "B-PER",   # Beginning of a person's name right after another person's name
    "I-PER",   # Person's name
    "B-ORG",   # Beginning of an organisation right after another organisation
    "I-ORG",   # Organisation
    "B-LOC",   # Beginning of a location right after another location
    "I-LOC"    # Location
]

sequence = "Hugging Face Inc. is a company based in New York City. Its headquarters are in DUMBO, therefore very" \
           "close to the Manhattan Bridge."

# Bit of a hack to get the tokens with the special tokens
tokens = tokenizer.tokenize(tokenizer.decode(tokenizer.encode(sequence)))
inputs = tokenizer.encode(sequence, return_tensors="pt")

outputs = model(inputs).logits
predictions = torch.argmax(outputs, dim=2)

print([(token, label_list[prediction]) for token, prediction in zip(tokens, predictions[0].numpy())])

===> [('[CLS]', 'O'), ('Hu', 'I-ORG'), ('##gging', 'I-ORG'), ('Face', 'I-ORG'), ('Inc', 'I-ORG'), ('.', 'O'), ('is', 'O'), ('a', 'O'), ('company', 'O'), ('based', 'O'), ('in', 'O'), ('New', 'I-LOC'), ('York', 'I-LOC'), ('City', 'I-LOC'), ('.', 'O'), ('Its', 'O'), ('headquarters', 'O'), ('are', 'O'), ('in', 'O'), ('D', 'I-LOC'), ('##UM', 'I-LOC'), ('##BO', 'I-LOC'), (',', 'O'), ('therefore', 'O'), ('very', 'O'), ('##c', 'O'), ('##lose', 'O'), ('to', 'O'), ('the', 'O'), ('Manhattan', 'I-LOC'), ('Bridge', 'I-LOC'), ('.', 'O'), ('[SEP]', 'O')]

```

### 文本摘要
以下是使用pipeline的示例
```
from transformers import pipeline

summarizer = pipeline("summarization")

ARTICLE = """ New York (CNN)When Liana Barrientos was 23 years old, she got married in Westchester County, New York.
    A year later, she got married again in Westchester County, but to a different man and without divorcing her first husband.
    Only 18 days after that marriage, she got hitched yet again. Then, Barrientos declared "I do" five more times, sometimes only within two weeks of each other.
    In 2010, she married once more, this time in the Bronx. In an application for a marriage license, she stated it was her "first and only" marriage.
    Barrientos, now 39, is facing two criminal counts of "offering a false instrument for filing in the first degree," referring to her false statements on the
    2010 marriage license application, according to court documents.
    Prosecutors said the marriages were part of an immigration scam.
    On Friday, she pleaded not guilty at State Supreme Court in the Bronx, according to her attorney, Christopher Wright, who declined to comment further.
    After leaving court, Barrientos was arrested and charged with theft of service and criminal trespass for allegedly sneaking into the New York subway through an emergency exit, said Detective
    Annette Markowski, a police spokeswoman. In total, Barrientos has been married 10 times, with nine of her marriages occurring between 1999 and 2002.
    All occurred either in Westchester County, Long Island, New Jersey or the Bronx. She is believed to still be married to four men, and at one time, she was married to eight men at once, prosecutors say.
    Prosecutors said the immigration scam involved some of her husbands, who filed for permanent residence status shortly after the marriages.
    Any divorces happened only after such filings were approved. It was unclear whether any of the men will be prosecuted.
    The case was referred to the Bronx District Attorney\'s Office by Immigration and Customs Enforcement and the Department of Homeland Security\'s
    Investigation Division. Seven of the men are from so-called "red-flagged" countries, including Egypt, Turkey, Georgia, Pakistan and Mali.
    Her eighth husband, Rashid Rajput, was deported in 2006 to his native Pakistan after an investigation by the Joint Terrorism Task Force.
    If convicted, Barrientos faces up to four years in prison.  Her next court appearance is scheduled for May 18.
    """

print(summarizer(ARTICLE, max_length=130, min_length=30, do_sample=False))

===> [{'summary_text': 'Liana Barrientos, 39, is charged with two counts of "offering a false instrument for filing in the first degree" In total, she has been married 10 times, with nine of her marriages occurring between 1999 and 2002. She is believed to still be married to four men.'}]
```

基本过程如下：
- 首先实例化模型和分词器
- 定义输入文本
- 添加特定于T5的前缀"summarize: "
- 使用PreTrainedModel.generate()方法生成摘要
```
from transformers import AutoModelWithLMHead, AutoTokenizer

model = AutoModelWithLMHead.from_pretrained("t5-base")
tokenizer = AutoTokenizer.from_pretrained("t5-base")

# T5 uses a max_length of 512 so we cut the article to 512 tokens.
inputs = tokenizer.encode("summarize: " + ARTICLE, return_tensors="pt", max_length=512)
outputs = model.generate(inputs, max_length=150, min_length=40, length_penalty=2.0, num_beams=4, early_stopping=True)
```

### 翻译
以下是使用Pipeline进行翻译的示例
```
from transformers import pipeline

translator = pipeline("translation_en_to_de")
print(translator("Hugging Face is a technology company based in New York and Paris", max_length=40))

===> [{'translation_text': 'Hugging Face ist ein Technologieunternehmen mit Sitz in New York und Paris.'}]
```

基本步骤
- 首先实例化模型和分词器
- 定义输入文本
- 添加特定于T5的前缀"translate English to German: "
- 使用PreTrainedModel.generate()方法生成翻译
```
from transformers import AutoModelWithLMHead, AutoTokenizer

model = AutoModelWithLMHead.from_pretrained("t5-base")
tokenizer = AutoTokenizer.from_pretrained("t5-base")

inputs = tokenizer.encode("translate English to German: Hugging Face is a technology company based in New York and Paris", return_tensors="pt")
outputs = model.generate(inputs, max_length=40, num_beams=4, early_stopping=True)

print(tokenizer.decode(outputs[0]))

===> Hugging Face ist ein Technologieunternehmen mit Sitz in New York und Paris.
```