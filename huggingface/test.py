from transformers import BertTokenizer
tokenizer = BertTokenizer.from_pretrained("bert-base-cased")

sequence = "A Titan RTX has 24GB of VRAM"
sequence2 = "Where is HuggingFace based?"

inputs = tokenizer(sequence, sequence2)

print(inputs)