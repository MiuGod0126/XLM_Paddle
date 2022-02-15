import paddle
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

x=paddle.randn((3,5))
index=[0,1]
value=-1
out=index_fill(x,index,value,axis=1)
print(x,out)
## gather
x=paddle.randn((3,4))
index=paddle.to_tensor([[0, 0], [1, 0],[1,0]])
print(x)
print(gather(x,axis=1,index=index))