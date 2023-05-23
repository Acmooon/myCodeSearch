# Copyright (c) Microsoft Corporation. 
# Licensed under the MIT license.

import torch
import torch.nn as nn
from transformers import RobertaTokenizer, RobertaModel, RobertaConfig

# unixcoder 继承自 nn.Module
class UniXcoder(nn.Module):
    def __init__(self, model_name):
        """
            Build UniXcoder.

            Parameters:

            * `model_name`- huggingface model card name. e.g. microsoft/unixcoder-base
        """        
        # 调用父类的构造函数
        super(UniXcoder, self).__init__()
        # 通过model_name加载tokenizer
        self.tokenizer = RobertaTokenizer.from_pretrained(model_name)
        # 通过model_name加载模型的配置
        self.config = RobertaConfig.from_pretrained(model_name)
        self.config.is_decoder = True
        # 通过model_name和config加载模型
        self.model = RobertaModel.from_pretrained(model_name, config=self.config)
        
        # 在内存中定义一个常量bias，同时，模型保存和加载的时候可以写入和读出。
        # 这里的bias是一个下三角矩阵，用于掩盖decoder的输入
        # 例如，当decoder的输入为[1,2,3,4,5]时，经过bias后，变为[1,0,0,0,0]
        # 这样做的目的是，让decoder只能看到前一个token，而不能看到后面的token，使其更好的学习到前一个token的信息
        self.register_buffer("bias", torch.tril(torch.ones((1024, 1024), dtype=torch.uint8)).view(1,1024, 1024))
        # 通过nn.Linear定义一个全连接层，输入维度为hidden_size，输出维度为vocab_size
        # 这里的vocab_size是模型的词表大小，即模型可以识别的单词的数量，这里是50265
        # hidden_size是模型的隐藏层维度，这里是768
        # bias=False表示不使用偏置
        self.lm_head = nn.Linear(self.config.hidden_size, self.config.vocab_size, bias=False)
        # 将模型的词嵌入层的权重赋值给lm_head，这样做的目的是，让lm_head的权重和模型的词嵌入层的权重共享
        self.lm_head.weight = self.model.embeddings.word_embeddings.weight
        # 定义一个softmax层，dim=-1表示对最后一维进行softmax，即对每个单词的概率进行softmax，得到每个单词的概率分布
        self.lsm = nn.LogSoftmax(dim=-1)
        # 给模型的分词器添加<mask0>特殊标记，这个标记是用于掩盖decoder的输入
        self.tokenizer.add_tokens(["<mask0>"],special_tokens=True)
        
    
    def tokenize(self, inputs, mode="<encoder-only>", max_length=512, padding=False):
        """ 
        Convert string to token ids 
                
        Parameters:

        * `inputs`- list of input strings.
        * `max_length`- The maximum total source sequence length after tokenization.
        * `padding`- whether to pad source sequence length to max_length. 
        * `mode`- which mode the sequence will use. i.e. <encoder-only>, <decoder-only>, <encoder-decoder>
        """
        # assert断言 mode 和 maxlength
        assert mode in ["<encoder-only>", "<decoder-only>", "<encoder-decoder>"]
        assert max_length < 1024
        # 注入分词器
        tokenizer = self.tokenizer
        # 定义一个空列表，用于存放token的id
        tokens_ids = []
        for x in inputs:
            tokens = tokenizer.tokenize(x)
            if mode == "<encoder-only>":
                tokens = tokens[:max_length-4]
                tokens = [tokenizer.cls_token,mode,tokenizer.sep_token] + tokens + [tokenizer.sep_token]                
            elif mode == "<decoder-only>":
                tokens = tokens[-(max_length-3):]
                tokens = [tokenizer.cls_token,mode,tokenizer.sep_token] + tokens
            else:
                tokens = tokens[:max_length-5]
                tokens = [tokenizer.cls_token,mode,tokenizer.sep_token] + tokens + [tokenizer.sep_token]
            # 将token转换为id
            tokens_id = tokenizer.convert_tokens_to_ids(tokens)
            if padding:
                tokens_id = tokens_id + [self.config.pad_token_id] * (max_length-len(tokens_id))
            tokens_ids.append(tokens_id)
        return tokens_ids
            
    def forward(self, source_ids):   
        """ Obtain token embeddings and sentence embeddings """
        # 通过模型获取token的嵌入和句子的嵌入
        # source_ids是输入的token的id
        # mask是一个掩盖矩阵，用于掩盖decoder的输入
        # mask的维度是[batch_size,seq_length]
        # mask的值是0或1，0表示掩盖，1表示不掩盖
        mask = source_ids.ne(self.config.pad_token_id)
        # 通过模型获取token的嵌入和句子的嵌入
        # token_embeddings的维度是[batch_size,seq_length,hidden_size]
        # sentence_embeddings的维度是[batch_size,hidden_size]
        token_embeddings = self.model(source_ids,attention_mask = mask.(1) * mask.unsqueeze(2))[0]
        # 通过lm_head获取token的概率分布
        # token_logits的维度是[batch_size,seq_length,vocab_size]
        sentence_embeddings = (token_embeddings * mask.unsqueeze(-1)).sum(1) / mask.sum(-1).unsqueeze(-1)
        return token_embeddings, sentence_embeddings       
