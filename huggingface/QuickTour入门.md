## QuickTour入门
### 1. 使用pipeline构建一个情感分类器
```
from transformers import pipeline

classifier = pipeline('sentiment-analysis')

# 对一个句子进行分类
classifier('We are very happy to show you the 🤗 Transformers library.')
# 对一堆句子进行分类
results = classifier(["We are very happy to show you the 🤗 Transformers library.",
                      "We hope you don't hate it."])
```

### 2. 使用别的模型进行情感分类
首先需要选择一个预训练的模型类型，从预训练的模型配置中取出分词器和模型。最后使用pipeline构建模型。
```
from transformers import AutoTokenizer, AutoModelForSequenceClassification

model_name = "nlptown/bert-base-multilingual-uncased-sentiment"
model = AutoModelForSequenceClassification.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)
classifier = pipeline('sentiment-analysis', model=model, tokenizer=tokenizer)
```

### 3. 不使用pipeline进行分类
首先类似前面的操作，取出模型和分词器
```
from transformers import AutoTokenizer, AutoModelForSequenceClassification
model_name = "distilbert-base-uncased-finetuned-sst-2-english"
pt_model = AutoModelForSequenceClassification.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)
```
对输入进行分词
```
inputs = tokenizer("We are very happy to show you the 🤗 Transformers library.")
```
类似于上一行，如果是对一大堆数据进行分词，可以得到一个数据batch。这里可以处理padding和截断，注意返回的时候指定模型类型pt还是tf。
```
pt_batch = tokenizer(
    ["We are very happy to show you the 🤗 Transformers library.", "We hope you don't hate it."],
    padding=True,
    truncation=True,
    return_tensors="pt")
```
然后将上面的batch传入到模型中。注意对于pytorch的版本需要使用两个星号\*\*来unpack上一步得到的输入。
```
pt_outputs = pt_model(**pt_batch)
```
然后还可以在前面的输出结果上加进去激活函数
```
import torch.nn.functional as F
pt_predictions = F.softmax(pt_outputs[0], dim=-1)
```
如果在前面调用模型的时候给出目标label的值，还可以得到损失和最后的pt_outputs结果：
```
import torch
pt_outputs = pt_model(**pt_batch, labels = torch.tensor([1, 0]))
```
### 4. 模型的保存
记得保存分词器和模型
```
tokenizer.save_pretrained(save_directory)
model.save_pretrained(save_directory)
```
### 5. 模型的加载
加载的时候如果是从另一个后端训练得到的模型，加载Model时要进行指明
```
tokenizer = AutoTokenizer.from_pretrained(save_directory)
model = TFAutoModel.from_pretrained(save_directory, from_pt=True)
```
或
```
tokenizer = AutoTokenizer.from_pretrained(save_directory)
model = AutoModel.from_pretrained(save_directory, from_tf=True)
```
### 6. 输出所有隐藏层和注意力
可以让模型输出所有隐藏层和注意力权重
```
pt_outputs = pt_model(**pt_batch, output_hidden_states=True, output_attentions=True)
all_hidden_states, all_attentions = pt_outputs[-2:]
```

### 7. 定制模型
可以通过Config文件来修改模型的一些参数。我个人的理解这里应该是由两种Config，一种是基类Config，通常而言修改基类Config的内容不会影响到模型的结构，因此修改基类Config的时候可以使用通用的pretrained得到的模型；
但是对于非基类的Config，例如特定的DistilBertConfig，修改这类Config中的参数会涉及模型结构的改变，就需要从零开始训练一个模型。

从零开始构建模型的代码
```
from transformers import DistilBertConfig, DistilBertTokenizer, DistilBertForSequenceClassification
config = DistilBertConfig(n_heads=8, dim=512, hidden_dim=4*512)
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
model = DistilBertForSequenceClassification(config)
```

只修改BaseConfig因此还可以使用预训练模型的代码
```
from transformers import DistilBertConfig, DistilBertTokenizer, DistilBertForSequenceClassification
model_name = "distilbert-base-uncased"
model = DistilBertForSequenceClassification.from_pretrained(model_name, num_labels=10)
tokenizer = DistilBertTokenizer.from_pretrained(model_name)
```