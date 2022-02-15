# coding=utf-8
# Copyright 2019-present, Facebook, Inc and the HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
 Paddle XLM model.
"""

'''
改为tokenizer后知道首先面对的困难是基类的区别，而model里面有许多算子需要修改，一步步来。【xlm参考bert和gpt】
先统统使用pretrainedmodel，然后修改XLMModel，再去修改其他task类。transformer的部分不要重复造轮子。
1.换掉paddle api,修改import：
    修改gelu；删除file_utils，modeling_outputs，modeling_utils；修改日志；删除XLMConfig，并到XLMPreTrainedModel类内修改config_class = XLMConfig为pretrained_init_configuration
    导入基类from .. import PretrainedModel, register_base_model,
    添加all,先空着.
2.修改类:
    让抽象类XLMPreTrainedModel继承
    发现有个尾缀为Output的类(XLMForQuestionAnsweringOutput),发现gpt无,bert有,参考Bert的paddle实现,发现Output类就是为了在某个任务类里作为输出的,集成在内即可,先不管  (可以删掉)
    把继承的Module改下.
    删除add_start_docstrings
    _prune_heads好像删掉了
    从上倒下画下所有类的继承关系,做到心中有数.(待会再调整类吧!!)
3.简单api翻译(先)
    view √
    transpose √
    .contiguous()√
    cat√ ,dim√,tensor√,
    .float() f32√
    FloatTensor√,
    long(torch.long->paddle.int64,torch.int->paddle.int32)√
    ,device√,size√ (函数:repeat,transpose),
    to√,type_as√,
    eps(LayerNorm)√
    
    其他不确定的函数:
    expand_as √ 不变
    masked_fill_ 自定义 √
    repeat-> tile √
    _init_weights √
    expand(x,shape) 不变√
    chunk不变 √
    index_select(dim,index)->index_select(index,axis) √
    # model_utils √
    # (待测试)adaptive softmax 还差gather，但是默认不走ada，asm=false  √
        new_zeros, paddle.zeros((b,),dtype=xx) # 和x.new_zeros相同类型 √
        new_empty  √
        nonzero https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/nonzero_cn.html#nonzero √
        (待测试)index_copy_ ,将copy tensor拼接到尾部，然后对索引修改后，重新选择tensor √
        (待测试)gather，如果二维的话用index_sample试试（一致），不是二维也要转到二维！√
        index_fill_(dim, index, value)-> √
        
    apply_chunking_to_forward,√
    find_pruneable_heads_and_indices,√
    prune_linear_layer√
                
4.从底层开始逐个理解,改写,测试,合并.直至跑通输出
    config输入这么改、output怎么输出。这两个搞完基本上可以
5.输出对齐
_ASMoutput = namedtuple('ASMoutput', ['output', 'loss']) # 名为ASMoutput的具名元组，有'output', 'loss'两个字段

'''
import itertools
import math
from dataclasses import dataclass
import numpy as np
import paddle
from paddle import nn
from paddle.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from paddlenlp.utils.log import logger
from paddle.nn.initializer import Constant,Normal,KaimingUniform,Uniform
from .. import (PretrainedModel,
                register_base_model,
                apply_chunking_to_forward,
                find_pruneable_heads_and_indices,
                prune_linear_layer)

__all__ = [
    'XLMModel',
    'XLMPreTrainedModel',
    'XLMWithLMHeadModel',
    'XLMForSequenceClassification',
    'XLMForQuestionAnsweringSimple',
    'XLMForQuestionAnswering',
    'XLMForTokenClassification',
    'XLMForMultipleChoice',
]

_CHECKPOINT_FOR_DOC = "xlm-mlm-en-2048"
_CONFIG_FOR_DOC = "XLMConfig"
_TOKENIZER_FOR_DOC = "XLMTokenizer"

XLM_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "xlm-mlm-en-2048",
    "xlm-mlm-ende-1024",
    "xlm-mlm-enfr-1024",
    "xlm-mlm-enro-1024",
    "xlm-mlm-tlm-xnli15-1024",
    "xlm-mlm-xnli15-1024",
    "xlm-clm-enfr-1024",
    "xlm-clm-ende-1024",
    "xlm-mlm-17-1280",
    "xlm-mlm-100-1280",
    # See all XLM models at https://huggingface.co/models?filter=xlm
]

def masked_fill(x, mask, value):
    # y = paddle.full(x.shape, value, x.dtype)
    return paddle.where(mask, paddle.to_tensor(value,dtype=x.dtype), x)

def index_fill(x,index,value,axis=0):
    ''' 生成一行掩码，然后按axis进行repeat，再用masked_fill'''
    index=paddle.to_tensor(index,dtype='int32') # int32

    mask = paddle.full([x.shape[axis]], 0, dtype='int64')
    mask[index]=1
    mask = paddle.cast(mask, dtype='bool')
    mask=mask.expand_as(x)

    out=masked_fill(x,mask,value)
    return out

def index_copy(input,copy_tensor,index,axis=0):
    ''' 将copy_tensor按index，在axis轴上覆盖input，除了axis外，其他维度一样'''
    shape_input=input.shape
    shape_copy=copy_tensor.shape
    axis_len=input.shape[axis]
    del shape_input[axis],shape_copy[axis]
    assert shape_input==shape_copy
    input=paddle.concat([input,copy_tensor],axis=axis)
    shift_index=index+axis_len
    final_index=paddle.arange(axis_len)
    final_index[index]=shift_index

    # 选sample
    out=paddle.index_select(input,index=final_index,axis=0)
    return out

def gather(x, axis, index):
    index_shape = index.shape
    index_flatten = index.flatten()
    if axis < 0:  # 最后一维
        axis = x.ndim + axis
    nd_index = []
    for k in range(x.ndim):
        if k == axis:
            nd_index.append(index_flatten)
        else:
            reshape_shape = [1] * x.ndim
            reshape_shape[k] = x.shape[k]
            dim_index = paddle.expand(paddle.arange(x.shape[k], dtype=index.dtype).reshape(reshape_shape),
                                      index_shape).flatten()
            nd_index.append(dim_index)
    paddle_out = paddle.gather_nd(x, paddle.stack(nd_index, axis=-1)).reshape(index_shape)
    return paddle_out

# -*- coding: utf-8 -*-

from collections import namedtuple



_ASMoutput = namedtuple('ASMoutput', ['output', 'loss']) # 名为ASMoutput的具名元组，有'output', 'loss'两个字段
class AdaptiveLogSoftmaxWithLoss(nn.Layer):
    r"""Efficient softmax approximation as described in
    `Efficient softmax approximation for GPUs by Edouard Grave, Armand Joulin,
    Moustapha Cissé, David Grangier, and Hervé Jégou
    <https://arxiv.org/abs/1609.04309>`__.

    Adaptive softmax is an approximate strategy for training models with large
    output spaces. It is most effective when the label distribution is highly
    imbalanced, for example in natural language modelling, where the word
    frequency distribution approximately follows the `Zipf's law`_.

    Adaptive softmax partitions the labels into several clusters, according to
    their frequency. These clusters may contain different number of targets
    each.
    Additionally, clusters containing less frequent labels assign lower
    dimensional embeddings to those labels, which speeds up the computation.
    For each minibatch, only clusters for which at least one target is
    present are evaluated.

    The idea is that the clusters which are accessed frequently
    (like the first one, containing most frequent labels), should also be cheap
    to compute -- that is, contain a small number of assigned labels.

    We highly recommend taking a look at the original paper for more details.

    * :attr:`cutoffs` should be an ordered Sequence of integers sorted
      in the increasing order.
      It controls number of clusters and the partitioning of targets into
      clusters. For example setting ``cutoffs = [10, 100, 1000]``
      means that first `10` targets will be assigned
      to the 'head' of the adaptive softmax, targets `11, 12, ..., 100` will be
      assigned to the first cluster, and targets `101, 102, ..., 1000` will be
      assigned to the second cluster, while targets
      `1001, 1002, ..., n_classes - 1` will be assigned
      to the last, third cluster.

    * :attr:`div_value` is used to compute the size of each additional cluster,
      which is given as
      :math:`\left\lfloor\frac{\texttt{in\_features}}{\texttt{div\_value}^{idx}}\right\rfloor`,
      where :math:`idx` is the cluster index (with clusters
      for less frequent words having larger indices,
      and indices starting from :math:`1`).

    * :attr:`head_bias` if set to True, adds a bias term to the 'head' of the
      adaptive softmax. See paper for details. Set to False in the official
      implementation.

    .. warning::
        Labels passed as inputs to this module should be sorted according to
        their frequency. This means that the most frequent label should be
        represented by the index `0`, and the least frequent
        label should be represented by the index `n_classes - 1`.

    .. note::
        This module returns a ``NamedTuple`` with ``output``
        and ``loss`` fields. See further documentation for details.

    .. note::
        To compute log-probabilities for all classes, the ``log_prob``
        method can be used.

    Args:
        in_features (int): Number of features in the input tensor
        n_classes (int): Number of classes in the dataset
        cutoffs (Sequence): Cutoffs used to assign targets to their buckets
        div_value (float, optional): value used as an exponent to compute sizes
            of the clusters. Default: 4.0
        head_bias (bool, optional): If ``True``, adds a bias term to the 'head' of the
            adaptive softmax. Default: ``False``

    Returns:
        ``NamedTuple`` with ``output`` and ``loss`` fields:
            * **output** is a Tensor of size ``N`` containing computed target
              log probabilities for each example
            * **loss** is a Scalar representing the computed negative
              log likelihood loss

    Shape:
        - input: :math:`(N, \texttt{in\_features})`
        - target: :math:`(N)` where each value satisfies :math:`0 <= \texttt{target[i]} <= \texttt{n\_classes}`
        - output1: :math:`(N)`
        - output2: ``Scalar``

    .. _Zipf's law: https://en.wikipedia.org/wiki/Zipf%27s_law
    """

    # in_features: int
    # n_classes: int
    # cutoffs: List[int]
    # div_value: float
    # head_bias: bool
    # head: Linear
    # tail: nn.LayerList

    def __init__(
        self,
        in_features,
        n_classes,
        cutoffs,
        div_value = 4.,
        head_bias= False
    ) -> None:
        super(AdaptiveLogSoftmaxWithLoss, self).__init__()

        cutoffs = list(cutoffs)

        if (cutoffs != sorted(cutoffs)) \
                or (min(cutoffs) <= 0) \
                or (max(cutoffs) > (n_classes - 1)) \
                or (len(set(cutoffs)) != len(cutoffs)) \
                or any([int(c) != c for c in cutoffs]):

            raise ValueError("cutoffs should be a sequence of unique, positive "
                             "integers sorted in an increasing order, where "
                             "each value is between 1 and n_classes-1")

        self.in_features = in_features
        self.n_classes = n_classes
        self.cutoffs = cutoffs + [n_classes]
        self.div_value = div_value
        self.head_bias = head_bias

        self.shortlist_size = self.cutoffs[0]
        self.n_clusters = len(self.cutoffs) - 1
        self.head_size = self.shortlist_size + self.n_clusters

        self.head = nn.Linear(self.in_features, self.head_size, bias_attr=self.head_bias)
        self.tail = nn.LayerList()

        for i in range(self.n_clusters):

            hsz = int(self.in_features // (self.div_value ** (i + 1)))
            osz = self.cutoffs[i + 1] - self.cutoffs[i]

            projection = nn.Sequential(
                nn.Linear(self.in_features, hsz, bias_attr=False),
                nn.Linear(hsz, osz, bias_attr=False)
            )

            self.tail.append(projection)

    def _calculate_fan_in_and_fan_out(self,tensor):
        dimensions = tensor.dim()
        if dimensions < 2:
            raise ValueError("Fan in and fan out can not be computed for tensor with fewer than 2 dimensions")

        num_input_fmaps = tensor.shape[1]
        num_output_fmaps = tensor.shape[0]
        receptive_field_size = 1
        if tensor.dim() > 2:
            receptive_field_size = tensor[0][0].numel()
        fan_in = num_input_fmaps * receptive_field_size
        fan_out = num_output_fmaps * receptive_field_size

        return fan_in, fan_out

    def reset_linear_parameters(self,layer) -> None:
        kaiming_uniform_=KaimingUniform(math.sqrt(5))
        kaiming_uniform_(layer.weight)
        if layer.bias is not None:
            fan_in, _ = self._calculate_fan_in_and_fan_out(layer.weight)
            bound = 1 / math.sqrt(fan_in)
            uniform_ = Uniform(low=-bound,high=bound)
            uniform_(layer.bias)

    def reset_parameters(self) -> None:
        self.reset_linear_parameters(self.head)
        for i2h, h2o in self.tail:
            self.reset_linear_parameters(i2h)
            self.reset_linear_parameters(h2o)

    def forward(self, input, target) -> _ASMoutput:
        if input.shape[0] != target.shape[0]:
            raise RuntimeError('Input and target should have the same size '
                               'in the batch dimension.')

        used_rows = 0
        batch_size = target.shape[0]
        output=paddle.zeros((batch_size,),dtype=input.dtype)
        gather_inds = paddle.empty((batch_size,),dtype=target.dtype)

        cutoff_values = [0] + self.cutoffs
        for i in range(len(cutoff_values) - 1):

            low_idx = cutoff_values[i]
            high_idx = cutoff_values[i + 1]

            target_mask = (target >= low_idx) & (target < high_idx)
            row_indices = target_mask.nonzero().squeeze()

            if row_indices.numel() == 0:
                continue

            if i == 0:
                gather_inds=index_copy(gather_inds,target[target_mask],index=row_indices,axis=0)
                # gather_inds.index_copy_(0, row_indices, target[target_mask])

            else:
                relative_target = target[target_mask] - low_idx
                input_subset = input.index_select(index=row_indices,axis=0)

                cluster_output = self.tail[i - 1](input_subset)
                cluster_index = self.shortlist_size + i - 1

                gather_inds=index_fill(gather_inds,index=row_indices,value=cluster_index,axis=0)
                # gather_inds.index_fill_(0, row_indices, cluster_index) # dim index value

                cluster_logprob = nn.functional.log_softmax(cluster_output, axis=1)
                local_logprob = gather(cluster_logprob,axis=1,index=relative_target.unsqueeze(1))
                # local_logprob = cluster_logprob.gather(1, relative_target.unsqueeze(1)) # .
                # output.index_copy_(0, row_indices, local_logprob.squeeze(1))
                output=index_copy(output,local_logprob.squeeze(1),index=row_indices,axis=0)

            used_rows += row_indices.numel()

        if used_rows != batch_size:
            raise RuntimeError("Target values should be in [0, {}], "
                               "but values in range [{}, {}] "
                               "were found. ".format(self.n_classes - 1,
                                                     target.min().item(),
                                                     target.max().item()))

        head_output = self.head(input)
        head_logprob = nn.functional.log_softmax(head_output, axis=1)
        output += gather(head_logprob,axis=1, index=gather_inds.unsqueeze(1)).squeeze()
        # output += head_logprob.gather(1, gather_inds.unsqueeze(1)).squeeze() # .
        loss = (-output).mean()

        return _ASMoutput(output, loss)

    def _get_full_log_prob(self, input, head_output):
        """ Given input tensor, and output of `self.head`,
        compute the log of the full distribution """

        out = paddle.empty((head_output.shape[0], self.n_classes),dtype=input.dtype)
        head_logprob =  nn.functional.log_softmax(head_output, axis=1)

        out[:, :self.shortlist_size] = head_logprob[:, :self.shortlist_size]

        for i, (start_idx, stop_idx) in enumerate(zip(self.cutoffs, self.cutoffs[1:])):
            cluster_output = self.tail[i](input)
            cluster_logprob = nn.functional.log_softmax(cluster_output, axis=1)
            output_logprob = cluster_logprob + head_logprob[:, self.shortlist_size + i].unsqueeze(1)

            out[:, start_idx:stop_idx] = output_logprob

        return out

    def log_prob(self, input):
        r""" Computes log probabilities for all :math:`\texttt{n\_classes}`

        Args:
            input (Tensor): a minibatch of examples

        Returns:
            log-probabilities of for each class :math:`c`
            in range :math:`0 <= c <= \texttt{n\_classes}`, where :math:`\texttt{n\_classes}` is a
            parameter passed to ``AdaptiveLogSoftmaxWithLoss`` constructor.

        Shape:
            - Input: :math:`(N, \texttt{in\_features})`
            - Output: :math:`(N, \texttt{n\_classes})`

        """

        head_output = self.head(input)
        return self._get_full_log_prob(input, head_output)

    def predict(self, input):
        r""" This is equivalent to `self.log_pob(input).argmax(dim=1)`,
        but is more efficient in some cases.

        Args:
            input (Tensor): a minibatch of examples

        Returns:
            output (Tensor): a class with the highest probability for each example

        Shape:
            - Input: :math:`(N, \texttt{in\_features})`
            - Output: :math:`(N)`
        """

        head_output = self.head(input)
        output = paddle.argmax(head_output, axis=1)
        not_in_shortlist = (output >= self.shortlist_size)
        all_in_shortlist = not (not_in_shortlist.any())

        if all_in_shortlist:
            return output

        elif not_in_shortlist.all():
            log_prob = self._get_full_log_prob(input, head_output)
            return paddle.argmax(log_prob, axis=1)

        else:
            log_prob = self._get_full_log_prob(input[not_in_shortlist],
                                               head_output[not_in_shortlist])
            output[not_in_shortlist] = paddle.argmax(log_prob, axis=1)
            return output


def create_sinusoidal_embeddings(n_pos, dim, out):
    position_enc = np.array([[pos / np.power(10000, 2 * (j // 2) / dim) for j in range(dim)] for pos in range(n_pos)])
    out[:, 0::2] = paddle.to_tensor(np.sin(position_enc[:, 0::2]),dtype='float32')
    out[:, 1::2] = paddle.to_tensor(np.cos(position_enc[:, 1::2]),dtype='float32')
    out=out.detach()
    out.stop_gradient = True


def get_masks(slen, lengths, causal, padding_mask=None):
    """
    Generate hidden states mask, and optionally an attention mask.
    """
    alen = paddle.arange(slen, dtype='int64')
    if padding_mask is not None:
        mask = padding_mask
    else:
        assert lengths.max().item() <= slen
        mask = alen < lengths[:, None]

    # attention mask is the same as mask, or triangular inferior attention (causal)
    bs = lengths.shape[0]
    if causal:
        attn_mask = alen[None, None, :].tile((bs, slen, 1)) <= alen[None, :, None]
    else:
        attn_mask = mask

    # sanity check
    assert mask.shape== [bs, slen]
    assert causal is False or attn_mask.shape == [bs, slen, slen]

    return mask, attn_mask


class MultiHeadAttention(nn.Layer):

    NEW_ID = itertools.count()

    def __init__(self, n_heads, dim, config):
        super().__init__()
        self.layer_id = next(MultiHeadAttention.NEW_ID)
        self.dim = dim
        self.n_heads = n_heads
        self.dropout = config.attention_dropout
        assert self.dim % self.n_heads == 0

        self.q_lin = nn.Linear(dim, dim)
        self.k_lin = nn.Linear(dim, dim)
        self.v_lin = nn.Linear(dim, dim)
        self.out_lin = nn.Linear(dim, dim)
        self.pruned_heads = set()

    def prune_heads(self, heads):
        attention_head_size = self.dim // self.n_heads
        if len(heads) == 0:
            return
        heads, index = find_pruneable_heads_and_indices(heads, self.n_heads, attention_head_size, self.pruned_heads)
        # Prune linear layers
        self.q_lin = prune_linear_layer(self.q_lin, index)
        self.k_lin = prune_linear_layer(self.k_lin, index)
        self.v_lin = prune_linear_layer(self.v_lin, index)
        self.out_lin = prune_linear_layer(self.out_lin, index, axis=1)
        # Update hyper params
        self.n_heads = self.n_heads - len(heads)
        self.dim = attention_head_size * self.n_heads
        self.pruned_heads = self.pruned_heads.union(heads)

    def forward(self, input, mask, kv=None, cache=None, head_mask=None, output_attentions=False):
        """
        Self-attention (if kv is None) or attention over source sentence (provided by kv).
        """
        # Input is (bs, qlen, dim)
        # Mask is (bs, klen) (non-causal) or (bs, klen, klen)
        bs, qlen, dim = input.shape
        if kv is None:
            klen = qlen if cache is None else cache["slen"] + qlen
        else:
            klen = kv.shape[1]
        # assert dim == self.dim, f'Dimensions do not match: {dim} input vs {self.dim} configured'
        n_heads = self.n_heads
        dim_per_head = self.dim // n_heads
        mask_reshape = (bs, 1, qlen, klen) if mask.dim() == 3 else (bs, 1, 1, klen)

        def shape(x):
            """projection"""
            return x.reshape((bs, -1, self.n_heads, dim_per_head)).transpose((0,2,1,3))

        def unshape(x):
            """compute context"""
            return x.transpose((0,2,1,3)).reshape((bs, -1, self.n_heads * dim_per_head))

        q = shape(self.q_lin(input))  # (bs, n_heads, qlen, dim_per_head)
        if kv is None:
            k = shape(self.k_lin(input))  # (bs, n_heads, qlen, dim_per_head)
            v = shape(self.v_lin(input))  # (bs, n_heads, qlen, dim_per_head)
        elif cache is None or self.layer_id not in cache:
            k = v = kv
            k = shape(self.k_lin(k))  # (bs, n_heads, qlen, dim_per_head)
            v = shape(self.v_lin(v))  # (bs, n_heads, qlen, dim_per_head)

        if cache is not None:
            if self.layer_id in cache:
                if kv is None:
                    k_, v_ = cache[self.layer_id]
                    k = paddle.concat([k_, k], axis=2)  # (bs, n_heads, klen, dim_per_head)
                    v = paddle.concat([v_, v], axis=2)  # (bs, n_heads, klen, dim_per_head)
                else:
                    k, v = cache[self.layer_id]
            cache[self.layer_id] = (k, v)

        q = q / math.sqrt(dim_per_head)  # (bs, n_heads, qlen, dim_per_head)
        scores = paddle.matmul(q, k.transpose((0,1,3,2)))  # (bs, n_heads, qlen, klen)
        mask = (mask == 0).reshape(mask_reshape).expand_as(scores)  # (bs, n_heads, qlen, klen)
        scores=masked_fill(scores,mask,value=-1e-9) # (bs, n_heads, qlen, klen)

        weights = paddle.cast(nn.functional.softmax(paddle.cast(scores,dtype='float32'), axis=-1),dtype=scores.dtype)# (bs, n_heads, qlen, klen)
        weights = nn.functional.dropout(weights, p=self.dropout, training=self.training)  # (bs, n_heads, qlen, klen)

        # Mask heads if we want to
        if head_mask is not None:
            weights = weights * head_mask

        context = paddle.matmul(weights, v)  # (bs, n_heads, qlen, dim_per_head)
        context = unshape(context)  # (bs, qlen, dim)

        outputs = (self.out_lin(context),)
        if output_attentions:
            outputs = outputs + (weights,)
        return outputs


class TransformerFFN(nn.Layer):
    def __init__(self, in_dim, dim_hidden, out_dim, config):
        super().__init__()
        self.dropout = config.dropout
        self.lin1 = nn.Linear(in_dim, dim_hidden)
        self.lin2 = nn.Linear(dim_hidden, out_dim)
        self.act = nn.functional.gelu if config.gelu_activation else nn.functional.relu
        self.chunk_size_feed_forward = config.chunk_size_feed_forward
        self.seq_len_dim = 1

    def forward(self, input):
        return apply_chunking_to_forward(self.ff_chunk, self.chunk_size_feed_forward, self.seq_len_dim, input)

    def ff_chunk(self, input):
        x = self.lin1(input)
        x = self.act(x)
        x = self.lin2(x)
        x = nn.functional.dropout(x, p=self.dropout, training=self.training)
        return x


class XLMPreTrainedModel(PretrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """
    # config_class = XLMConfig
    pretrained_init_configuration={}
    load_tf_weights = None
    base_model_prefix = "transformer"

    def __init__(self, *inputs, **kwargs):
        super().__init__(*inputs, **kwargs)

    @property
    def dummy_inputs(self):
        inputs_list = paddle.to_tensor([[7, 6, 0, 0, 1], [1, 2, 3, 0, 0], [0, 0, 0, 4, 5]])
        attns_list = paddle.to_tensor([[1, 1, 0, 0, 1], [1, 1, 1, 0, 0], [1, 0, 0, 1, 1]])
        if self.config.use_lang_emb and self.config.n_langs > 1:
            langs_list = paddle.to_tensor([[1, 1, 0, 0, 1], [1, 1, 1, 0, 0], [1, 0, 0, 1, 1]])
        else:
            langs_list = None
        return {"input_ids": inputs_list, "attention_mask": attns_list, "langs": langs_list}

    def _init_weights(self, module):
        """Initialize the weights."""
        normal_embed_=Normal(mean=0,std=self.config.embed_init_std)
        normal_init_=Normal(mean=0,std=self.config.init_std)
        ones_=Constant(value=1.)
        zeros_=Constant(value=0.)
        if isinstance(module, nn.Embedding):
            if self.config is not None and self.config.embed_init_std is not None:
                normal_embed_(module.weight)
            if module.padding_idx is not None:
                w=module.weight.detach()
                w[module.padding_idx]=0.
                module.weight.set_value(w)
        if isinstance(module, nn.Linear):
            if self.config is not None and self.config.init_std is not None:
                normal_init_(module.weight)
                if module.bias is not None:
                    zeros_(module.bias)
        if isinstance(module, nn.LayerNorm):
            zeros_(module.bias)
            ones_(module.weight)

@register_base_model
class XLMModel(XLMPreTrainedModel):
    _keys_to_ignore_on_load_missing = [r"position_ids"]

    def __init__(self, config):
        super().__init__(config)

        # encoder / decoder, output layer
        self.is_encoder = config.is_encoder
        self.is_decoder = not config.is_encoder
        if self.is_decoder:
            raise NotImplementedError("Currently XLM can only be used as an encoder")
        # self.with_output = with_output
        self.causal = config.causal

        # dictionary / languages
        self.n_langs = config.n_langs
        self.use_lang_emb = config.use_lang_emb
        self.n_words = config.n_words
        self.eos_index = config.eos_index
        self.pad_index = config.pad_index
        # self.dico = dico
        # self.id2lang = config.id2lang
        # self.lang2id = config.lang2id
        # assert len(self.dico) == self.n_words
        # assert len(self.id2lang) == len(self.lang2id) == self.n_langs

        # model parameters
        self.dim = config.emb_dim  # 512 by default
        self.hidden_dim = self.dim * 4  # 2048 by default
        self.n_heads = config.n_heads  # 8 by default
        self.n_layers = config.n_layers
        self.dropout = config.dropout
        self.attention_dropout = config.attention_dropout
        assert self.dim % self.n_heads == 0, "transformer dim must be a multiple of n_heads"

        # embeddings
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, self.dim)
        if config.sinusoidal_embeddings:
            create_sinusoidal_embeddings(config.max_position_embeddings, self.dim, out=self.position_embeddings.weight)
        if config.n_langs > 1 and config.use_lang_emb:
            self.lang_embeddings = nn.Embedding(self.n_langs, self.dim)
        self.embeddings = nn.Embedding(self.n_words, self.dim, padding_idx=self.pad_index)
        self.layer_norm_emb = nn.LayerNorm(self.dim, epsilon=config.layer_norm_eps)

        # transformer layers
        self.attentions = nn.LayerList()
        self.layer_norm1 = nn.LayerList()
        self.ffns = nn.LayerList()
        self.layer_norm2 = nn.LayerList()
        # if self.is_decoder:
        #     self.layer_norm15 = nn.LayerList()
        #     self.encoder_attn = nn.LayerList()

        for _ in range(self.n_layers):
            self.attentions.append(MultiHeadAttention(self.n_heads, self.dim, config=config))
            self.layer_norm1.append(nn.LayerNorm(self.dim, epsilon=config.layer_norm_eps))
            # if self.is_decoder:
            #     self.layer_norm15.append(nn.LayerNorm(self.dim, epsilon=config.layer_norm_eps))
            #     self.encoder_attn.append(MultiHeadAttention(self.n_heads, self.dim, dropout=self.attention_dropout))
            self.ffns.append(TransformerFFN(self.dim, self.hidden_dim, self.dim, config=config))
            self.layer_norm2.append(nn.LayerNorm(self.dim, epsilon=config.layer_norm_eps))

        if hasattr(config, "pruned_heads"):
            pruned_heads = config.pruned_heads.copy().items()
            config.pruned_heads = {}
            for layer, heads in pruned_heads:
                if self.attentions[int(layer)].n_heads == config.n_heads:
                    self.prune_heads({int(layer): list(map(int, heads))})


        # Initialize weights and apply final processing
        self.post_init()
        self.register_buffer("position_ids", paddle.arange(config.max_position_embeddings).expand((1, -1)))

    def get_input_embeddings(self):
        return self.embeddings

    def set_input_embeddings(self, new_embeddings):
        self.embeddings = new_embeddings

    def _prune_heads(self, heads_to_prune):
        """
        Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer} See base
        class PreTrainedModel
        """
        for layer, heads in heads_to_prune.items():
            self.attentions[layer].prune_heads(heads)

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        langs=None,
        token_type_ids=None,
        position_ids=None,
        lengths=None,
        cache=None,
        head_mask=None,
        inputs_embeds=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if input_ids is not None:
            bs, slen = input_ids.shape
        else:
            bs, slen = inputs_embeds.shape[:-1]

        # device = input_ids.device if input_ids is not None else inputs_embeds.device

        if lengths is None:
            if input_ids is not None:
                lengths = paddle.cast(paddle.sum(input_ids != self.pad_index,axis=1),dtype='int64')
                # lengths = (input_ids != self.pad_index).sum(dim=1).long()
            else:
                lengths = paddle.to_tensor([slen] * bs)
        # mask = input_ids != self.pad_index

        # check inputs
        assert lengths.shape[0] == bs
        assert lengths.max().item() <= slen
        # input_ids = input_ids.transpose(0, 1)  # batch size as dimension 0
        # assert (src_enc is None) == (src_len is None)
        # if src_enc is not None:
        #     assert self.is_decoder
        #     assert src_enc.shape[0] == bs

        # generate masks
        mask, attn_mask = get_masks(slen, lengths, self.causal, padding_mask=attention_mask)
        # if self.is_decoder and src_enc is not None:
        #     src_mask = paddle.arange(src_len.max(), dtype=paddle.long, device=lengths.device) < src_len[:, None]

        # position_ids
        if position_ids is None:
            position_ids = self.position_ids[:, :slen]
        else:
            assert position_ids.shape == [bs, slen]  # [slen, bs]
            # position_ids = position_ids.transpose(0, 1)

        # langs
        if langs is not None:
            assert langs.shape == [bs, slen]  # [slen, bs]
            # langs = langs.transpose(0, 1)

        # Prepare head mask if needed
        head_mask = self.get_head_mask(head_mask, self.config.n_layers)

        # do not recompute cached elements
        if cache is not None and input_ids is not None:
            _slen = slen - cache["slen"]
            input_ids = input_ids[:, -_slen:]
            position_ids = position_ids[:, -_slen:]
            if langs is not None:
                langs = langs[:, -_slen:]
            mask = mask[:, -_slen:]
            attn_mask = attn_mask[:, -_slen:]

        # embeddings
        if inputs_embeds is None:
            inputs_embeds = self.embeddings(input_ids)

        tensor = inputs_embeds + self.position_embeddings(position_ids).expand_as(inputs_embeds)
        if langs is not None and self.use_lang_emb and self.n_langs > 1:
            tensor = tensor + self.lang_embeddings(langs)
        if token_type_ids is not None:
            tensor = tensor + self.embeddings(token_type_ids)
        tensor = self.layer_norm_emb(tensor)
        tensor = nn.functional.dropout(tensor, p=self.dropout, training=self.training)
        tensor *= paddle.cast(mask.unsqueeze(-1), dtype=tensor.dtype)

        # transformer layers
        hidden_states = () if output_hidden_states else None
        attentions = () if output_attentions else None
        for i in range(self.n_layers):
            if output_hidden_states:
                hidden_states = hidden_states + (tensor,)

            # self attention
            attn_outputs = self.attentions[i](
                tensor,
                attn_mask,
                cache=cache,
                head_mask=head_mask[i],
                output_attentions=output_attentions,
            )
            attn = attn_outputs[0]
            if output_attentions:
                attentions = attentions + (attn_outputs[1],)
            attn = nn.functional.dropout(attn, p=self.dropout, training=self.training)
            tensor = tensor + attn
            tensor = self.layer_norm1[i](tensor)

            # encoder attention (for decoder only)
            # if self.is_decoder and src_enc is not None:
            #     attn = self.encoder_attn[i](tensor, src_mask, kv=src_enc, cache=cache)
            #     attn = nn.functional.dropout(attn, p=self.dropout, training=self.training)
            #     tensor = tensor + attn
            #     tensor = self.layer_norm15[i](tensor)

            # FFN
            tensor = tensor + self.ffns[i](tensor)
            tensor = self.layer_norm2[i](tensor)
            tensor *= paddle.cast(mask.unsqueeze(-1),dtype=tensor.dtype)

        # Add last hidden state
        if output_hidden_states:
            hidden_states = hidden_states + (tensor,)

        # update cache length
        if cache is not None:
            cache["slen"] += tensor.shape[1]

        # move back sequence length to dimension 0
        # tensor = tensor.transpose(0, 1)

        if not return_dict:
            return tuple(v for v in [tensor, hidden_states, attentions] if v is not None)
        return BaseModelOutput(last_hidden_state=tensor, hidden_states=hidden_states, attentions=attentions)


class XLMPredLayer(nn.Layer):
    """
    Prediction layer (cross_entropy or adaptive_softmax).
    """

    def __init__(self, config):
        super().__init__()
        self.asm = config.asm
        self.n_words = config.n_words
        self.pad_index = config.pad_index
        dim = config.emb_dim

        if config.asm is False:
            self.proj = nn.Linear(dim, config.n_words, bias_attr=True)
        else:
            self.proj = AdaptiveLogSoftmaxWithLoss(
                in_features=dim,
                n_classes=config.n_words,
                cutoffs=config.asm_cutoffs,
                div_value=config.asm_div_value,
                head_bias=True,  # default is False
            )

    def forward(self, x, y=None):
        """Compute the loss, and optionally the scores."""
        outputs = ()
        if self.asm is False:
            scores = self.proj(x)
            outputs = (scores,) + outputs
            if y is not None:
                loss = nn.functional.cross_entropy(
                    scores.reshape((-1, self.n_words)), y.reshape((-1)), reduction="elementwise_mean"
                )
                outputs = (loss,) + outputs
        else:
            scores = self.proj.log_prob(x)
            outputs = (scores,) + outputs
            if y is not None:
                _, loss = self.proj(x, y)
                outputs = (loss,) + outputs

        return outputs


class XLMWithLMHeadModel(XLMPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.transformer = XLMModel(config)
        self.pred_layer = XLMPredLayer(config)

        # Initialize weights and apply final processing
        self.post_init()

    def get_output_embeddings(self):
        return self.pred_layer.proj

    def set_output_embeddings(self, new_embeddings):
        self.pred_layer.proj = new_embeddings

    def prepare_inputs_for_generation(self, input_ids, **kwargs):
        mask_token_id = self.config.mask_token_id
        lang_id = self.config.lang_id

        effective_batch_size = input_ids.shape[0]
        mask_token = paddle.full((effective_batch_size, 1), mask_token_id, dtype='int64')
        input_ids = paddle.concat([input_ids, mask_token], axis=1)
        if lang_id is not None:
            langs = paddle.full_like(input_ids, lang_id)
        else:
            langs = None
        return {"input_ids": input_ids, "langs": langs}

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        langs=None,
        token_type_ids=None,
        position_ids=None,
        lengths=None,
        cache=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        r"""
        labels (:obj:`paddle.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
            Labels for language modeling. Note that the labels **are shifted** inside the model, i.e. you can set
            ``labels = input_ids`` Indices are selected in ``[-100, 0, ..., config.vocab_size]`` All labels set to
            ``-100`` are ignored (masked), the loss is only computed for labels in ``[0, ..., config.vocab_size]``
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        transformer_outputs = self.transformer(
            input_ids,
            attention_mask=attention_mask,
            langs=langs,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            lengths=lengths,
            cache=cache,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        output = transformer_outputs[0]
        outputs = self.pred_layer(output, labels)  # (loss, logits) or (logits,) depending on if labels are provided.

        if not return_dict:
            return outputs + transformer_outputs[1:]

        return MaskedLMOutput(
            loss=outputs[0] if labels is not None else None,
            logits=outputs[0] if labels is None else outputs[1],
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
        )



class XLMForSequenceClassification(XLMPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.config = config

        self.transformer = XLMModel(config)
        self.sequence_summary = SequenceSummary(config)

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        langs=None,
        token_type_ids=None,
        position_ids=None,
        lengths=None,
        cache=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        r"""
        labels (:obj:`paddle.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
            Labels for computing the sequence classification/regression loss. Indices should be in :obj:`[0, ...,
            config.num_labels - 1]`. If :obj:`config.num_labels == 1` a regression loss is computed (Mean-Square loss),
            If :obj:`config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        transformer_outputs = self.transformer(
            input_ids,
            attention_mask=attention_mask,
            langs=langs,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            lengths=lengths,
            cache=cache,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        output = transformer_outputs[0]
        logits = self.sequence_summary(output)

        loss = None
        if labels is not None:
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == paddle.int64 or labels.dtype == paddle.int32):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.reshape((-1, self.num_labels)), labels.reshape((-1)))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)

        if not return_dict:
            output = (logits,) + transformer_outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
        )


class XLMForQuestionAnsweringSimple(XLMPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        self.transformer = XLMModel(config)
        self.qa_outputs = nn.Linear(config.hidden_size, config.num_labels)

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        langs=None,
        token_type_ids=None,
        position_ids=None,
        lengths=None,
        cache=None,
        head_mask=None,
        inputs_embeds=None,
        start_positions=None,
        end_positions=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        r"""
        start_positions (:obj:`paddle.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
            Labels for position (index) of the start of the labelled span for computing the token classification loss.
            Positions are clamped to the length of the sequence (:obj:`sequence_length`). Position outside of the
            sequence are not taken into account for computing the loss.
        end_positions (:obj:`paddle.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
            Labels for position (index) of the end of the labelled span for computing the token classification loss.
            Positions are clamped to the length of the sequence (:obj:`sequence_length`). Position outside of the
            sequence are not taken into account for computing the loss.
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        transformer_outputs = self.transformer(
            input_ids,
            attention_mask=attention_mask,
            langs=langs,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            lengths=lengths,
            cache=cache,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = transformer_outputs[0]

        logits = self.qa_outputs(sequence_output)
        start_logits, end_logits = logits.split(1, axis=-1)
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)

        total_loss = None
        if start_positions is not None and end_positions is not None:
            # If we are on multi-GPU, split add a dimension
            if len(start_positions.shape) > 1:
                start_positions = start_positions.squeeze(-1)
            if len(end_positions.shape) > 1:
                end_positions = end_positions.squeeze(-1)
            # sometimes the start/end positions are outside our model inputs, we ignore these terms
            ignored_index = start_logits.shape[1]
            start_positions = start_positions.clamp(0, ignored_index)
            end_positions = end_positions.clamp(0, ignored_index)

            loss_fct = CrossEntropyLoss(ignore_index=ignored_index)
            start_loss = loss_fct(start_logits, start_positions)
            end_loss = loss_fct(end_logits, end_positions)
            total_loss = (start_loss + end_loss) / 2

        if not return_dict:
            output = (start_logits, end_logits) + transformer_outputs[1:]
            return ((total_loss,) + output) if total_loss is not None else output

        return QuestionAnsweringModelOutput(
            loss=total_loss,
            start_logits=start_logits,
            end_logits=end_logits,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
        )


class XLMForQuestionAnswering(XLMPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        self.transformer = XLMModel(config)
        self.qa_outputs = SQuADHead(config)

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        langs=None,
        token_type_ids=None,
        position_ids=None,
        lengths=None,
        cache=None,
        head_mask=None,
        inputs_embeds=None,
        start_positions=None,
        end_positions=None,
        is_impossible=None,
        cls_index=None,
        p_mask=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        r"""
        start_positions (:obj:`paddle.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
            Labels for position (index) of the start of the labelled span for computing the token classification loss.
            Positions are clamped to the length of the sequence (:obj:`sequence_length`). Position outside of the
            sequence are not taken into account for computing the loss.
        end_positions (:obj:`paddle.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
            Labels for position (index) of the end of the labelled span for computing the token classification loss.
            Positions are clamped to the length of the sequence (:obj:`sequence_length`). Position outside of the
            sequence are not taken into account for computing the loss.
        is_impossible (``paddle.LongTensor`` of shape ``(batch_size,)``, `optional`):
            Labels whether a question has an answer or no answer (SQuAD 2.0)
        cls_index (``paddle.LongTensor`` of shape ``(batch_size,)``, `optional`):
            Labels for position (index) of the classification token to use as input for computing plausibility of the
            answer.
        p_mask (``paddle.FloatTensor`` of shape ``(batch_size, sequence_length)``, `optional`):
            Optional mask of tokens which can't be in answers (e.g. [CLS], [PAD], ...). 1.0 means token should be
            masked. 0.0 mean token is not masked.

        Returns:

        Example::

            # >>> from transformers import XLMTokenizer, XLMForQuestionAnswering
            # >>> import paddle
            #
            # >>> tokenizer = XLMTokenizer.from_pretrained('xlm-mlm-en-2048')
            # >>> model = XLMForQuestionAnswering.from_pretrained('xlm-mlm-en-2048')
            #
            # >>> input_ids = paddle.to_tensor(tokenizer.encode("Hello, my dog is cute", add_special_tokens=True)).unsqueeze(0)  # Batch size 1
            # >>> start_positions = paddle.to_tensor([1])
            # >>> end_positions = paddle.to_tensor([3])
            #
            # >>> outputs = model(input_ids, start_positions=start_positions, end_positions=end_positions)
            # >>> loss = outputs.loss
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        transformer_outputs = self.transformer(
            input_ids,
            attention_mask=attention_mask,
            langs=langs,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            lengths=lengths,
            cache=cache,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        output = transformer_outputs[0]

        outputs = self.qa_outputs(
            output,
            start_positions=start_positions,
            end_positions=end_positions,
            cls_index=cls_index,
            is_impossible=is_impossible,
            p_mask=p_mask,
            return_dict=return_dict,
        )

        if not return_dict:
            return outputs + transformer_outputs[1:]

        return XLMForQuestionAnsweringOutput(
            loss=outputs.loss,
            start_top_log_probs=outputs.start_top_log_probs,
            start_top_index=outputs.start_top_index,
            end_top_log_probs=outputs.end_top_log_probs,
            end_top_index=outputs.end_top_index,
            cls_logits=outputs.cls_logits,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
        )


class XLMForTokenClassification(XLMPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.transformer = XLMModel(config)
        self.dropout = nn.Dropout(config.dropout)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        langs=None,
        token_type_ids=None,
        position_ids=None,
        lengths=None,
        cache=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        r"""
        labels (:obj:`paddle.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
            Labels for computing the token classification loss. Indices should be in ``[0, ..., config.num_labels -
            1]``.
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.transformer(
            input_ids,
            attention_mask=attention_mask,
            langs=langs,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            lengths=lengths,
            cache=cache,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]

        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)

        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            # Only keep active parts of the loss
            if attention_mask is not None:
                active_loss = attention_mask.reshape((-1)) == 1
                active_logits = logits.reshape((-1, self.num_labels))
                active_labels = paddle.where(
                    active_loss, labels.reshape((-1)), paddle.to_tensor(loss_fct.ignore_index,dtype=labels.dtype)
                )
                loss = loss_fct(active_logits, active_labels)
            else:
                loss = loss_fct(logits.reshape((-1, self.num_labels)), labels.reshape((-1)))

        if not return_dict:
            output = (logits,) + outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return TokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

class XLMForMultipleChoice(XLMPreTrainedModel):
    def __init__(self, config, *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)

        self.transformer = XLMModel(config)
        self.sequence_summary = SequenceSummary(config)
        self.logits_proj = nn.Linear(config.num_labels, 1)

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        langs=None,
        token_type_ids=None,
        position_ids=None,
        lengths=None,
        cache=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        r"""
        labels (:obj:`paddle.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
            Labels for computing the multiple choice classification loss. Indices should be in ``[0, ...,
            num_choices-1]`` where :obj:`num_choices` is the size of the second dimension of the input tensors. (See
            :obj:`input_ids` above)
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        num_choices = input_ids.shape[1] if input_ids is not None else inputs_embeds.shape[1]

        input_ids = input_ids.reshape((-1, input_ids.shape[-1])) if input_ids is not None else None
        attention_mask = attention_mask.reshape((-1, attention_mask.shape[-1])) if attention_mask is not None else None
        token_type_ids = token_type_ids.reshape((-1, token_type_ids.shape[-1])) if token_type_ids is not None else None
        position_ids = position_ids.reshape((-1, position_ids.shape[-1])) if position_ids is not None else None
        langs = langs.reshape((-1, langs.shape[-1])) if langs is not None else None
        inputs_embeds = (
            inputs_embeds.reshape((-1, inputs_embeds.shape[-2], inputs_embeds.shape[-1]))
            if inputs_embeds is not None
            else None
        )

        if lengths is not None:
            logger.warning(
                "The `lengths` parameter cannot be used with the XLM multiple choice models. Please use the "
                "attention mask instead."
            )
            lengths = None

        transformer_outputs = self.transformer(
            input_ids=input_ids,
            attention_mask=attention_mask,
            langs=langs,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            lengths=lengths,
            cache=cache,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        output = transformer_outputs[0]
        logits = self.sequence_summary(output)
        logits = self.logits_proj(logits)
        reshaped_logits = logits.reshape((-1, num_choices))

        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(reshaped_logits, labels)

        if not return_dict:
            output = (reshaped_logits,) + transformer_outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return MultipleChoiceModelOutput(
            loss=loss,
            logits=reshaped_logits,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
        )
