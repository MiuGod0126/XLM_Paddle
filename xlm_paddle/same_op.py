import paddle
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

input=paddle.randn((5,3))
cpy=paddle.randn((3,3))
index=paddle.to_tensor([0,1,2])

print('input',input)
print('cpy',cpy)
out=index_copy(input,cpy,index,axis=0)
print('out',out)