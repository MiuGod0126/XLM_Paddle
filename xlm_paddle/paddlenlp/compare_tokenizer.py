'''
对齐xlm tokenizer的tokenize，tostring,...
'''
from paddlenlp.transformers import XLMTokenizer
tokenizer=XLMTokenizer.from_pretrained('xlm-mlm-en-2048') # xlm可能放成对句子，也要测试
s1='Faith can move mountains. '
out_paddle=tokenizer(s1)
input_ids=out_paddle['input_ids']
new_str=tokenizer.convert_ids_to_string(input_ids)
print('paddle tokenized:',out_paddle)
print('paddle str:',new_str)

from transformers import XLMTokenizer  as TCXLMTokenizer
tokenizer_torch=TCXLMTokenizer.from_pretrained('xlm-mlm-en-2048')
out_torch=tokenizer_torch(s1)
print('torch tokenized:',out_torch)
torch_str=tokenizer_torch.convert_tokens_to_string([tokenizer_torch._convert_id_to_token(ids) for ids in input_ids])
print('torch str:',torch_str)


########### 测试PADDLE convert demo ##############
raw_str='Welcome to use PaddlePaddle and PaddleNLP'
print('raw str:',raw_str)
tokenized=tokenizer(raw_str)# tokenized: {'input_ids': [0, 4848, 21, 253, 4277, 24154, 25754, 18, 4277, 212, 3220, 7534, 1], 'token_type_ids': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]}
print('tokenized:',tokenized)
ids=tokenized['input_ids']
## convert_ids_to_tokens
tokens=tokenizer.convert_ids_to_tokens(ids)
print('tokens:',tokens) #tokens: ['<s>', 'welcome</w>', 'to</w>', 'use</w>', 'pad', 'dle', 'paddle</w>', 'and</w>', 'pad', 'd', 'len', 'lp</w>', '</s>']
## convert_tokens_to_ids
re_ids=tokenizer.convert_tokens_to_ids(tokens)
print('re_ids:',re_ids) # re_ids: [0, 4848, 21, 253, 4277, 24154, 25754, 18, 4277, 212, 3220, 7534, 1]
## convert_ids_to_string
re_str=tokenizer.convert_ids_to_string(re_ids)
print('new str:',re_str) # new str: <s>welcome to use paddlepaddle and paddlenlp </s>
