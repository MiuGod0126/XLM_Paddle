from transformers import XLMTokenizer, XLMForSequenceClassification,XLMModel
import torch
from transformers import XLMConfig, XLMModel
# Initializing a XLM configuration
configuration = XLMConfig()
# Initializing a model from the configuration
model = XLMForSequenceClassification(configuration)

tokenizer = XLMTokenizer.from_pretrained('xlm-mlm-en-2048')
# model = XLMForSequenceClassification.from_pretrained('xlm-mlm-en-2048', return_dict=True)
inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")
labels = torch.tensor([1]).unsqueeze(0)  # Batch size 1
outputs = model(**inputs, labels=labels)
loss = outputs.loss
logits = outputs.logits
print(logits)
# 到site packages里面去找