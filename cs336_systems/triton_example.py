import triton
import triton.language as tl
import torch
from jaxtyping import Float
from torch import Tensor
from einops import rearrange

@triton.jit
def weighted_sum_fwd(
    x_ptr,              # (ROWS,D)
    w_ptr,              # (D,)
    o_ptr,              # (ROWS,)
    x_stride_rows,
    x_stride_d,
    w_stride_d,
    o_stride_rows,
    ROWS, D,
    ROWS_TILE_SIZE: tl.constexpr, D_TILE_SIZE: tl.constexpr
):
    # Grid.shape = (cdiv(ROWS, ROWS_TILE_SIZE),)
    rows_program_id = tl.program_id(0)

    # compute block ptr
    x_block_ptr = tl.make_block_ptr(
        x_ptr,
        shape=(ROWS,D),
        strides=(x_stride_rows,x_stride_d),
        offsets=(rows_program_id * ROWS_TILE_SIZE, 0),
        block_shape=(ROWS_TILE_SIZE,D_TILE_SIZE),
        order=(1,0)
    )

    w_block_ptr = tl.make_block_ptr(
        w_ptr,
        shape=(D,),
        strides=(w_stride_d,),
        offsets=(0,),
        block_shape=(D_TILE_SIZE,),
        order=(0,)
    )

    o_block_ptr = tl.make_block_ptr(
        o_ptr,
        shape=(ROWS,),
        strides=(o_stride_rows,),
        offsets=(rows_program_id * ROWS_TILE_SIZE,),
        block_shape=(ROWS_TILE_SIZE,),
        order=(0,)
    )

    output = tl.zeros((ROWS_TILE_SIZE,), dtype = tl.float32)

    for i in range (tl.cdiv(D, D_TILE_SIZE)):
        # load data
        x = tl.load(x_block_ptr, boundary_check=(0,1), padding_option="zero")
        w = tl.load(w_block_ptr, boundary_check=(0,), padding_option="zero")

        # compute
        output += tl.sum((x * w[None,:]), axis=1)

        # advance x and w
        x_block_ptr = x_block_ptr.advance((0, D_TILE_SIZE))
        w_block_ptr = w_block_ptr.advance((D_TILE_SIZE,))

    # store
    tl.store(o_block_ptr, output, boundary_check=(0,))

@triton.jit
def weighted_sum_bwd(
    output_grad_ptr,        #(ROWS,)    input
    x_ptr,                  #(ROWS,D)   input
    w_ptr,                  #(D,)       input
    x_grad_ptr,             #(ROWS,D)   output
    w_partial_grad_ptr,     #(NUM_ROWS_TILE,D)       output
    output_grad_stride_rows,
    x_stride_rows, x_stride_d,
    w_stride_d,
    x_grad_stride_rows, x_grad_stride_d,
    w_grad_stride_rows, w_grad_stride_d,
    ROWS, D,
    ROWS_TILE_SIZE: tl.constexpr, D_TILE_SIZE: tl.constexpr
):
    program_id = tl.program_id(0)
    NUM_ROWS_TILE = tl.num_programs(0)

    output_grad_block_ptr = tl.make_block_ptr(
        output_grad_ptr,
        shape=(ROWS,),
        strides=(output_grad_stride_rows,),
        offsets=(program_id * ROWS_TILE_SIZE,),
        block_shape=(ROWS_TILE_SIZE,),
        order=(0,)
    )

    x_block_ptr = tl.make_block_ptr(
        x_ptr,
        shape=(ROWS,D),
        strides=(x_stride_rows, x_stride_d),
        offsets=(program_id * ROWS_TILE_SIZE, 0),
        block_shape=(ROWS_TILE_SIZE, D_TILE_SIZE),
        order=(1,0)
    )

    w_block_ptr = tl.make_block_ptr(
        w_ptr,
        shape=(D,),
        strides=(w_stride_d,),
        offsets=(0,),
        block_shape=(D_TILE_SIZE,),
        order=(0,)
    )

    x_grad_block_ptr = tl.make_block_ptr(
        x_grad_ptr,
        shape=(ROWS,D),
        strides=(x_grad_stride_rows, x_grad_stride_d),
        offsets=(program_id * ROWS_TILE_SIZE, 0),
        block_shape=(ROWS_TILE_SIZE, D_TILE_SIZE),
        order=(1,0)
    )

    w_partial_grad_block_ptr = tl.make_block_ptr(
        w_partial_grad_ptr,
        shape=(NUM_ROWS_TILE, D),
        strides=(w_grad_stride_rows,w_grad_stride_d),
        offsets=(program_id,0),
        block_shape=(1, D_TILE_SIZE),
        order=(1,0)
    )
    output_grad = tl.load(output_grad_block_ptr, boundary_check=(0,), padding_option="zero")    # (ROWS_TILE_SIZE,1)
    for i in range(tl.cdiv(ROWS, ROWS_TILE_SIZE)):
        # compute x_grad
        w = tl.load(w_block_ptr, boundary_check=(0,), padding_option="zero")
        x_grad = output_grad[:, None] * w[None, :]
        tl.store(x_grad_block_ptr, x_grad, boundary_check=(0,1))
        x_grad_block_ptr = x_grad_block_ptr.advance((0, D_TILE_SIZE))

        # compute w_grad
        x = tl.load(x_block_ptr, boundary_check=(0,1), padding_option="zero")                             # (ROWS_TILE_SIZE,D_TILE_SIZE)
        w_partial_grad = tl.sum((x * output_grad[:, None]), axis=0, keep_dims = True)
        tl.store(w_partial_grad_block_ptr, w_partial_grad, boundary_check=(1,))
        w_partial_grad_block_ptr = w_partial_grad_block_ptr.advance((0, D_TILE_SIZE))

        w_block_ptr = w_block_ptr.advance((D_TILE_SIZE,))
        x_block_ptr = x_block_ptr.advance((0, D_TILE_SIZE))

        


class WeightedSum(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: Float[Tensor, "... D"], w: Float[Tensor, "D"]) -> Float[Tensor, "..."]:
        assert x.is_cuda and w.is_cuda
        assert len(w.shape) == 1
        assert x.shape[-1] == w.shape[0]

        ctx.save_for_backward(x, w)
        
        w_dim, output_dim = x.shape[-1], x.shape[:-1]

        ctx.ROWS_TILE_SIZE = 16
        ctx.D_TILE_SIZE = triton.next_power_of_2(w_dim) // 16
        ctx.input_shape = x.shape

        output = torch.empty(output_dim, device=x.device)
        
        x = rearrange(x, "... D -> (...) D")
        output = rearrange(output, "... -> (...)")

        n_rows = output.numel()
        weighted_sum_fwd[(triton.cdiv(n_rows, ctx.ROWS_TILE_SIZE),)](
            x,w,output,
            x.stride(0),x.stride(1),
            w.stride(0),
            output.stride(0),
            ROWS = n_rows, D = w_dim,
            ROWS_TILE_SIZE = ctx.ROWS_TILE_SIZE, D_TILE_SIZE = ctx.D_TILE_SIZE
        )

        return output.view(output_dim)
    
    @staticmethod
    def backward(ctx, output_grad:Float[Tensor, "..."]):
        assert output_grad.is_cuda
        device = output_grad.device
        ROWS_TILE_SIZE = ctx.ROWS_TILE_SIZE
        D_TILE_SIZE = ctx.D_TILE_SIZE
        input_shape = ctx.input_shape
        x, w = ctx.saved_tensors

        x_grad = torch.empty(input_shape, dtype=torch.float32, device=device)
        x_grad = rearrange(x_grad, "... D -> (...) D")
        num_rows,D = x_grad.shape[0], x_grad.shape[1]
        w_partial_grad = torch.empty((triton.cdiv(num_rows, ROWS_TILE_SIZE), D), dtype=torch.float32, device=device)

        output_grad = rearrange(output_grad, "... -> (...)")
        weighted_sum_bwd[(triton.cdiv(num_rows, ROWS_TILE_SIZE),)] (
            output_grad_ptr=output_grad,
            x_ptr=x, w_ptr=w,
            x_grad_ptr=x_grad,
            w_partial_grad_ptr=w_partial_grad,
            output_grad_stride_rows=output_grad.stride(0),
            x_stride_rows=x.stride(0),x_stride_d=x.stride(1),
            w_stride_d=w.stride(0),
            x_grad_stride_rows=x_grad.stride(0), x_grad_stride_d=x_grad.stride(1),
            w_grad_stride_rows=w_partial_grad.stride(0), w_grad_stride_d=w_partial_grad.stride(1),
            ROWS=num_rows, D = D,
            ROWS_TILE_SIZE=ROWS_TILE_SIZE,D_TILE_SIZE=D_TILE_SIZE
        )
        x_grad = x_grad.view(input_shape)
        w_grad = w_partial_grad.sum(dim=0)
        return x_grad, w_grad
        

def main():
    x = torch.rand((128, 1024),dtype=torch.float32,device="cuda",requires_grad=True)
    w = torch.rand((1024,), dtype=torch.float32, device="cuda",requires_grad=True)
    y = WeightedSum.apply(x,w)
    
    x_exp = x
    w_exp = w
    y_expected = torch.sum((x_exp * w_exp[None, :]), dim=1)
    assert all(abs(y - y_expected) < 1e-4)
    print(y)
    loss = y.reshape(-1).sum()
    loss.backward()
    loss_exp = y_expected.reshape(-1).sum()
    loss_exp.backward()

    assert all(abs(w.grad.data - w_exp.grad.data) < 1e-4)

    x = torch.rand((8,128, 1024),dtype=torch.float32,device="cuda")
    w = torch.rand((1024,), dtype=torch.float32, device="cuda")
    y = WeightedSum.apply(x,w)
    assert all((abs(y - torch.sum((x * w[None, None, :]), dim=2)) < 1e-4).reshape(-1))

if __name__ == '__main__':
    main()
