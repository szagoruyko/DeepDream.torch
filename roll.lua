require 'nvrtc'
require 'cutorch-rtc'

local ptx = nvrtc.compileReturnPTX[[
extern "C"
__global__ void roll(float *dst, const float *src, int rows, int cols, int y, int x)
{
    int tx = blockIdx.x*blockDim.x + threadIdx.x;
    int ty = blockIdx.y*blockDim.y + threadIdx.y;

    if(tx >= cols || ty >= rows)
        return;

    int dstx = (cols + (tx + x) % cols) % cols;
    int dsty = (rows + (ty + y) % rows) % rows;

    src += ty*cols + tx;
    dst += dsty*cols + dstx;

    dst[0] = src[0];
    dst[rows*cols] = src[rows*cols];
    dst[2*rows*cols] = src[2*rows*cols];
}
]]

function roll(dst,src,x,y)
    assert(src:isContiguous() and torch.isTypeOf(src,'torch.CudaTensor'))
    assert(dst:isContiguous() and torch.isTypeOf(dst,'torch.CudaTensor'))
    assert(src:isSameSizeAs(dst))
    
    
    local ch,rows,cols = unpack(src:size():totable())
    
    cutorch.launchPTX(ptx, 'roll', {
            dst,
            src,
            {'int', rows},
            {'int', cols},
            {'int', y},
            {'int', x}
        }, {math.ceil(cols/16)*16, math.ceil(rows/16)*16}, {16,16})
    return dst
end
