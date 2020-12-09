## Glossary术语

### 1. 文本到ID的变换
对于一个输入文本，使用tokenizer进行分词
```
from transformers import BertTokenizer
tokenizer = BertTokenizer.from_pretrained("bert-base-cased")

sequence = "A Titan RTX has 24GB of VRAM"
```
如果调用tokenizer的tokenize()方法可以得到一个分词后的数组。在BERT中对于没有见过的单词，会使用\#\#作为前缀表示这是个切出来的不完整单词
```
tokenized_sequence = tokenizer.tokenize(sequence)
print(tokenized_sequence)

===> ['A', 'Titan', 'R', '##T', '##X', 'has', '24', '##GB', 'of', 'V', '##RA', '##M']
```
如果是直接把文本传给tokenizer对象，会返回一个dict，其内容包括input_ids, token_type_ids, attention_mask三种数据。

其中，input_ids表明了句子分词后得到的对应idx序列。注意这里增加了句首的[CLS]项（id:101）和句尾的[SEP]项（id:201）。
```
encoded_sequence = inputs["input_ids"]
print(encoded_sequence)

===> [101, 138, 18696, 155, 1942, 3190, 1144, 1572, 13745, 1104, 159, 9664, 2107, 102]
```
对上述得到的idx重新解码，可以还原句子
```
decoded_sequence = tokenizer.decode(encoded_sequence)
print(decoded_sequence)

===> [CLS] A Titan RTX has 24GB of VRAM [SEP]
```
(嗯。。但是如果尝试把同一个句子不断input/output会在他外面包装很多个句首和句尾标记项 = =|||)

### 2. 注意力Mask
注意力Mask表明在整个batch里面哪一些词语需要被attention，哪一些可以直接忽略（通常他们是padding）。

例如，对于以下两个句子
```
from transformers import BertTokenizer
tokenizer = BertTokenizer.from_pretrained("bert-base-cased")

sequence_a = "This is a short sequence."
sequence_b = "This is a rather long sequence. It is at least longer than the sequence A."

encoded_sequence_a = tokenizer(sequence_a)["input_ids"]
encoded_sequence_b = tokenizer(sequence_b)["input_ids"]
```
编码得到两个句子的长度不同
```
len(encoded_sequence_a), len(encoded_sequence_b)

===> (8, 19)
```
由于长度不同的句子不能一起丢进同一个batch，需要进行padding
```
padded_sequences = tokenizer([sequence_a, sequence_b], padding=True)
```
padding之后查看attention_mask，就会发现它是一个和分词&编码后的数组维度一样大的矩阵，它表示每个位置是不是有意义的内容，如果是那就是1，如果是pad出来的那就是0。
```
padded_sequences["attention_mask"]

===> [[1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]]
```

### 3. Token类型ID
直观理解这是个用在具有两个句子的任务上的标记，比如说对于下面两个句子进行分词
```
from transformers import BertTokenizer
tokenizer = BertTokenizer.from_pretrained("bert-base-cased")
sequence_a = "HuggingFace is based in NYC"
sequence_b = "Where is HuggingFace based?"

encoded_dict = tokenizer(sequence_a, sequence_b)
decoded = tokenizer.decode(encoded_dict["input_ids"])
```
得到的分词结果是
```
 print(decoded)
===> [CLS] HuggingFace is based in NYC [SEP] Where is HuggingFace based? [SEP]
```
此时查看其token类型ID为
```
encoded_dict['token_type_ids']

===> [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1]
```
注意到，上述的最后一个0对应第一个句子末尾的[SEP]。

