import numpy as np
import torch
import torch.nn as nn
import math

def quantize(x, scale, zero, maxq):
    if maxq < 0:
        return (x > scale / 2).float() * scale + (x < zero / 2).float() * zero
    q = torch.clamp(torch.round(x / scale) + zero, 0, maxq)
    return scale * (q - zero)

class Quantizer(nn.Module):

    def __init__(self, shape=1):
        super(Quantizer, self).__init__()
        self.register_buffer('maxq', torch.tensor(0))
        self.register_buffer('scale', torch.zeros(shape))
        self.register_buffer('zero', torch.zeros(shape))

    def configure(
        self,
        bits, perchannel=False, sym=True,
        mse=False, norm=2.4, grid=100, maxshrink=.8,
        trits=False
        ):

        self.maxq = torch.tensor(2 ** bits - 1)
        self.perchannel = perchannel
        self.sym = sym
        self.mse = mse
        self.norm = norm
        self.grid = grid
        self.maxshrink = maxshrink
        if trits:
            self.maxq = torch.tensor(-1)

    def find_params(self, x, weight=False):
        dev = x.device
        self.maxq = self.maxq.to(dev)

        shape = x.shape
        if self.perchannel:
            if weight:
                x = x.flatten(1)
            else:
                if len(shape) == 4:
                    x = x.permute([1, 0, 2, 3])
                    x = x.flatten(1)
                if len(shape) == 3:
                    x = x.reshape((-1, shape[-1])).t()
                if len(shape) == 2:
                    x = x.t()
        else:
            x = x.flatten().unsqueeze(0)

        tmp = torch.zeros(x.shape[0], device=dev)
        xmin = torch.minimum(x.min(1)[0], tmp)
        xmax = torch.maximum(x.max(1)[0], tmp)

        if self.sym:
            xmax = torch.maximum(torch.abs(xmin), xmax)
            tmp = xmin < 0
            if torch.any(tmp):
                xmin[tmp] = -xmax[tmp]
        tmp = (xmin == 0) & (xmax == 0)
        xmin[tmp] = -1
        xmax[tmp] = +1

        if self.maxq < 0:
            self.scale = xmax
            self.zero = xmin
        else:
            self.scale = (xmax - xmin) / self.maxq
            if self.sym:
                self.zero = torch.full_like(self.scale, (self.maxq + 1) / 2)
            else:
                self.zero = torch.round(-xmin / self.scale)

        if self.mse:
            best = torch.full([x.shape[0]], float('inf'), device=dev)
            for i in range(int(self.maxshrink * self.grid)):
                p = 1 - i / self.grid
                xmin1 = p * xmin
                xmax1 = p * xmax
                scale1 = (xmax1 - xmin1) / self.maxq
                zero1 = torch.round(-xmin1 / scale1) if not self.sym else self.zero
                q = quantize(x, scale1.unsqueeze(1), zero1.unsqueeze(1), self.maxq)
                q -= x
                q.abs_()
                q.pow_(self.norm)
                err = torch.sum(q, 1)
                tmp = err < best
                if torch.any(tmp):
                    best[tmp] = err[tmp]
                    self.scale[tmp] = scale1[tmp]
                    self.zero[tmp] = zero1[tmp]
        if not self.perchannel:
            if weight:
                tmp = shape[0]
            else:
                tmp = shape[1] if len(shape) != 3 else shape[2]
            self.scale = self.scale.repeat(tmp)
            self.zero = self.zero.repeat(tmp)

        if weight:
            shape = [-1] + [1] * (len(shape) - 1)
            self.scale = self.scale.reshape(shape)
            self.zero = self.zero.reshape(shape)
            return
        if len(shape) == 4:
            self.scale = self.scale.reshape((1, -1, 1, 1))
            self.zero = self.zero.reshape((1, -1, 1, 1))
        if len(shape) == 3:
            self.scale = self.scale.reshape((1, 1, -1))
            self.zero = self.zero.reshape((1, 1, -1))
        if len(shape) == 2:
            self.scale = self.scale.unsqueeze(0)
            self.zero = self.zero.unsqueeze(0)

    def quantize(self, x):
        if self.ready():
            return quantize(x, self.scale, self.zero, self.maxq)
        return x

    def enabled(self):
        return self.maxq > 0

    def ready(self):
        return torch.all(self.scale != 0)


try:
    from . import quant_cuda
except:
    print('CUDA extension not installed.')

# Assumes layer is perfectly divisible into 256 * 256 blocks
class QuantLinear(nn.Module):
    def __init__(self, bits, groupsize, infeatures, outfeatures, bias, faster=False, kernel_switch_threshold=64):
        super().__init__()
        if bits not in [2,3,4,8]:
            raise NotImplementedError("Only 2,3,4,8 bits are supported.")
        self.infeatures = infeatures
        self.outfeatures = outfeatures
        self.bits = bits
        if groupsize != -1 and groupsize < 32 and groupsize != int(math.pow(2,int(math.log2(groupsize)))):
            raise NotImplementedError("groupsize supports powers of 2 greater than 32. (e.g. : 32,64,128,etc)")
        groupsize = groupsize if groupsize != -1 else infeatures
        self.groupsize = groupsize
        self.register_buffer('qzeros', torch.zeros((math.ceil(infeatures/groupsize),outfeatures // 256 * (bits * 8)), dtype=torch.int))
        self.register_buffer('scales', torch.zeros((math.ceil(infeatures/groupsize),outfeatures), dtype=torch.float16))
        if bias:
            self.register_buffer('bias', torch.zeros((outfeatures),dtype=torch.float16))
        else:
            self.bias = None
        self.register_buffer(
            'qweight', torch.zeros((infeatures // 32 * bits, outfeatures), dtype=torch.int)
        )
        self.register_buffer(
            'g_idx', torch.tensor([i // self.groupsize for i in range(infeatures)], dtype=torch.int)
        )
        self.half_indim = self.infeatures // 2
        self._initialized_quant_state = False
        self.faster = faster
        # kernel_switch_threshold is the cutoff input size after which matmul
        # is performed by unpacking the weights and using torch.matmul
        self.kernel_switch_threshold = kernel_switch_threshold
        if isinstance(self.kernel_switch_threshold, bool):
            self.kernel_switch_threshold = 128 if self.kernel_switch_threshold else None
        if not self.kernel_switch_threshold is None:
            # Buffers for bit shifting weight unpacking performed by unpacking the weights and using torch.matmul
            if self.bits in [2,4,8]:
                self.register_buffer('wf',torch.tensor(list(range(0,32,self.bits)), dtype=torch.int32).unsqueeze(0),persistent=False)
            elif self.bits == 3:
                self.register_buffer('wf', torch.tensor([[0, 3, 6, 9, 12, 15, 18, 21, 24, 27, 30, 0],
                                                        [0, 1, 4, 7, 10, 13, 16, 19, 22, 25, 28, 31],
                                                        [0, 2, 5, 8, 11, 14, 17, 20, 23, 26, 29, 0],], dtype=torch.int32).reshape(1,3,12), persistent=False)

    def pack(self, linear, scales, zeros, g_idx=None):
        self.g_idx = (g_idx.clone() if g_idx is not None else self.g_idx).to(torch.int)

        scales = scales.t().contiguous()
        zeros = zeros.t().contiguous()
        scale_zeros = zeros * scales
        self.scales = scales.clone()
        if linear.bias is not None:
            self.bias = linear.bias.clone()

        intweight = []
        for idx in range(self.infeatures):
            intweight.append(torch.round((linear.weight.data[:,idx] + scale_zeros[g_idx]) / self.scales[g_idx]).to(torch.int)[:,None])
        intweight = torch.cat(intweight,dim=1)
        intweight = intweight.t().contiguous()
        intweight = intweight.numpy().astype(np.uint32)
        qweight = np.zeros(
            (intweight.shape[0] // 32 * self.bits, intweight.shape[1]), dtype=np.uint32
        )
        i = 0
        row = 0
        while row < qweight.shape[0]:
            if self.bits in [2,4,8]:
                for j in range(i, i + (32//self.bits)):
                    qweight[row] |= intweight[j] << (self.bits * (j - i))
                i += 32//self.bits
                row += 1
            elif self.bits == 3:
                for j in range(i, i + 10):
                    qweight[row] |= intweight[j] << (3 * (j - i))
                i += 10
                qweight[row] |= intweight[i] << 30
                row += 1
                qweight[row] |= (intweight[i] >> 2) & 1
                i += 1
                for j in range(i, i + 10):
                    qweight[row] |= intweight[j] << (3 * (j - i) + 1)
                i += 10
                qweight[row] |= intweight[i] << 31
                row += 1
                qweight[row] |= (intweight[i] >> 1) & 0x3
                i += 1
                for j in range(i, i + 10):
                    qweight[row] |= intweight[j] << (3 * (j - i) + 2)
                i += 10
                row += 1
            else:
                raise NotImplementedError("Only 2,3,4,8 bits are supported.")

        qweight = qweight.astype(np.int32)
        self.qweight = torch.from_numpy(qweight)

        zeros -= 1;
        zeros = zeros.numpy().astype(np.uint32)
        qzeros = np.zeros((zeros.shape[0], zeros.shape[1] // 256 * (self.bits * 8)), dtype=np.uint32)
        i = 0
        col = 0
        while col < qzeros.shape[1]:
            if self.bits in [2,4,8]:
                for j in range(i, i + (32//self.bits)):
                    qzeros[:, col] |= zeros[:, j] << (self.bits * (j - i))
                i += 32//self.bits
                col += 1
            elif self.bits == 3:
                for j in range(i, i + 10):
                    qzeros[:, col] |= zeros[:, j] << (3 * (j - i))
                i += 10
                qzeros[:, col] |= zeros[:, i] << 30
                col += 1
                qzeros[:, col] |= (zeros[:, i] >> 2) & 1
                i += 1
                for j in range(i, i + 10):
                    qzeros[:, col] |= zeros[:, j] << (3 * (j - i) + 1)
                i += 10
                qzeros[:, col] |= zeros[:, i] << 31
                col += 1
                qzeros[:, col] |= (zeros[:, i] >> 1) & 0x3
                i += 1
                for j in range(i, i + 10):
                    qzeros[:, col] |= zeros[:, j] << (3 * (j - i) + 2)
                i += 10
                col += 1
            else:
                raise NotImplementedError("Only 2,3,4,8 bits are supported.")

        qzeros = qzeros.astype(np.int32)
        self.qzeros = torch.from_numpy(qzeros)

    def forward(self, x):
        if not self._initialized_quant_state:
            # Do we even have a bias? Check for at least one non-zero element.
            if self.bias is not None and bool(torch.any(self.bias != 0)):
                # Then make sure it's the right type.
                self.bias.data = self.bias.data.to(torch.float32)
            else:
                self.bias = None

        outshape = list(x.shape)
        outshape[-1] = self.outfeatures
        x = x.reshape(-1, x.shape[-1])
        if self.bias is None:
            y = torch.zeros(x.shape[0], outshape[-1], dtype=torch.float16 if self.faster else torch.float32, device=x.device)
        else:
            y = self.bias.clone().repeat(x.shape[0], 1)

        # from https://github.com/qwopqwop200/GPTQ-for-LLaMa/blob/cuda/quant.py
        # main kernel is very slow for big initial inputs
        if not self.kernel_switch_threshold is None and x.shape[0] > self.kernel_switch_threshold:
            if self.scales.dtype != torch.float16:
                self.scales = self.scales.half()
            if self.bits in [2,4,8]:
                zeros = torch.bitwise_right_shift(torch.unsqueeze(self.qzeros, 2).expand(-1, -1, 32 // self.bits), self.wf.unsqueeze(0)).to(torch.int16 if self.bits == 8 else torch.int8)
                torch.bitwise_and(zeros, (2 ** self.bits) - 1, out=zeros)

                zeros = zeros + 1
                zeros = zeros.reshape(self.scales.shape)

                weight = torch.bitwise_right_shift(torch.unsqueeze(self.qweight, 1).expand(-1, 32 // self.bits, -1), self.wf.unsqueeze(-1)).to(torch.int16 if self.bits == 8 else torch.int8)
                torch.bitwise_and(weight,(2 ** self.bits) - 1, out=weight)
            elif self.bits == 3:
                zeros = self.qzeros.reshape(self.qzeros.shape[0], self.qzeros.shape[1]//3, 3, 1).expand(-1, -1, -1, 12)
                zeros = (zeros >> self.wf.unsqueeze(0))
                zeros[:,:,0,10] = (zeros[:,:,0,10]&0x3) | ((zeros[:,:,1,0] << 2)&0x4)
                zeros[:,:,1,11] = (zeros[:,:,1,11]&0x1) | ((zeros[:,:,2,0] << 1)&0x6)
                zeros = zeros & 0x7
                zeros = torch.cat([zeros[:,:,0,:11], zeros[:,:,1,1:12], zeros[:,:,2,1:11]], dim=2)

                zeros = zeros + 1
                zeros = zeros.reshape(self.scales.shape)

                weight = self.qweight.reshape(self.qweight.shape[0]//3, 3, 1, self.qweight.shape[1]).expand(-1, -1, 12, -1)
                weight = (weight >> self.wf.unsqueeze(-1))&0x7
                weight[:,0,10] = (weight[:,0,10]&0x3) | ((weight[:,1,0] << 2)&0x4)
                weight[:,1,11] = (weight[:,1,11]&0x1) | ((weight[:,2,0] << 1)&0x6)
                weight = weight & 0x7
                weight = torch.cat([weight[:,0,:11], weight[:,1,1:12], weight[:,2,1:11]], dim=1)

            weight = weight.reshape(weight.shape[0] * weight.shape[1], weight.shape[2])
            num_itr = self.g_idx.shape[0]//x.shape[-1]
            if num_itr == 1:
                weights = (self.scales[self.g_idx.long()] * (weight - zeros[self.g_idx.long()]))
            else:
                num_dim = self.g_idx.shape[0]//num_itr
                weights = []
                for i in range(num_itr):
                    scale_i = self.scales[:,i*num_dim:(i+1)*num_dim]
                    weight_i = weight[:,i*num_dim:(i+1)*num_dim]
                    zeros_i = zeros[:,i*num_dim:(i+1)*num_dim]
                    g_idx_i = self.g_idx[i*num_dim:(i+1)*num_dim]
                    weights.append(scale_i[g_idx_i.long()] * (weight_i - zeros_i[g_idx_i.long()]))
                weights = torch.cat(weights,dim=1)
            out = torch.matmul(x.half(), weights)
            out_shape = x.shape[:-1] + (self.outfeatures, )
            out = out.reshape(out_shape)
            out = out + self.bias if self.bias is not None else out
            return out

        outshape = list(x.shape)
        outshape[-1] = self.outfeatures
        x = x.reshape(-1, x.shape[-1])
        if self.bias is None:
            y = torch.zeros(x.shape[0], outshape[-1], dtype=torch.float32, device=x.device)
        else:
            y = self.bias.clone().repeat(x.shape[0], 1)

        output_dtype = x.dtype
        if self.faster:
            x = x.half()
            if y.dtype != torch.float16:
                y = y.half()
            if self.bits == 3:
                quant_cuda.vecquant3matmul_faster(x, self.qweight, y, self.scales, self.qzeros, self.groupsize, self.half_indim)
            elif self.bits == 4:
                quant_cuda.vecquant4matmul_faster(x, self.qweight, y, self.scales, self.qzeros, self.g_idx, self.half_indim)
            else:
                raise NotImplementedError("Only 3 and 4 bits are supported.")
        else:
            x = x.float()
            if self.scales.dtype != torch.float32:
                self.scales = self.scales.float()
            if self.bits == 2:
                quant_cuda.vecquant2matmul(x, self.qweight, y, self.scales, self.qzeros, self.groupsize)
            elif self.bits == 3:
                quant_cuda.vecquant3matmul(x, self.qweight, y, self.scales, self.qzeros, self.groupsize)
            elif self.bits == 4:
                quant_cuda.vecquant4matmul(x, self.qweight, y, self.scales, self.qzeros, self.g_idx)
            elif self.bits == 8:
                quant_cuda.vecquant8matmul(x, self.qweight, y, self.scales, self.qzeros, self.groupsize)
            else:
                raise NotImplementedError("Only 2,3,4,8 bits are supported.")
        y = y.to(output_dtype)
        return y.reshape(outshape)

def make_quant(module, names, bits, groupsize, faster=True, name=''):
    if not faster and bits in [3,4]:
        faster=True
    if isinstance(module, QuantLinear):
        return
    for attr in dir(module):
        tmp = getattr(module, attr)
        name1 = name + '.' + attr if name != '' else attr
        if name1 in names:
            delattr(module, attr)
            setattr(module, attr, QuantLinear(bits, groupsize, tmp.in_features, tmp.out_features, tmp.bias is not None, faster=faster))
    for name1, child in module.named_children():
        make_quant(child, names, bits, groupsize, faster, name + '.' + name1 if name != '' else name1)