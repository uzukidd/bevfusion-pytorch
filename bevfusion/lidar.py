"""Inference-only sparse convolution reference backend for the legacy checkpoint.
Uses PyTorch CUDA gather/GEMM and preserves legacy [kx,ky,kz,in,out] weights.
No third-party sparse extensions, and no modification of upstream sources.
"""
from dataclasses import dataclass, field
from itertools import product
import time
import torch
from torch import nn
from torch.nn import functional as F


def triple(value):
    return (value,) * 3 if isinstance(value, int) else tuple(value)


def keys(coords, shape):
    return (coords[:, 0] * shape[1] + coords[:, 1]) * shape[2] + coords[:, 2]


@dataclass
class SparseState:
    features: torch.Tensor
    coords: torch.Tensor
    shape: tuple
    cache: dict = field(default_factory=dict)

    def replace(self, values):
        return SparseState(values, self.coords, self.shape, self.cache)


class ReferenceSparseConv(nn.Module):
    def __init__(self, cin, cout, kernel=3, stride=1, padding=0, subm=False):
        super().__init__()
        self.kernel, self.stride, self.padding = triple(kernel), triple(stride), triple(padding)
        self.subm = subm
        self.weight = nn.Parameter(torch.empty(*self.kernel, cin, cout))

    def forward(self, state):
        if state.features.shape[0] == 0:
            out_shape = state.shape if self.subm else tuple(
                (s + 2 * p - k) // st + 1
                for s, p, k, st in zip(state.shape, self.padding, self.kernel, self.stride)
            )
            return SparseState(
                state.features.new_zeros((0, self.weight.shape[-1])),
                state.coords.new_empty((0, 3)), out_shape,
                state.cache if self.subm else {},
            )
        device = state.features.device
        stride = torch.tensor(self.stride, device=device)
        # Legacy submanifold conv fixes the centre at kernel//2, independent of declared padding.
        pad_tuple = tuple(k // 2 for k in self.kernel) if self.subm else self.padding
        padding = torch.tensor(pad_tuple, device=device)
        offsets = list(product(*(range(k) for k in self.kernel)))
        cache_key = (self.kernel, self.stride, pad_tuple, self.subm)
        if self.subm and cache_key in state.cache:
            out_coords, out_shape, pairs = state.cache[cache_key]
        else:
            in_shape = state.shape
            if self.subm:
                out_coords, out_shape = state.coords, in_shape
            else:
                out_shape = tuple((s + 2*p - k)//st + 1 for s,p,k,st in zip(in_shape,self.padding,self.kernel,self.stride))
                candidates = []
                shape_tensor = torch.tensor(out_shape,device=device)
                for offset in offsets:
                    numerator = state.coords + padding - torch.tensor(offset,device=device)
                    valid = (numerator.remainder(stride) == 0).all(1)
                    co = numerator.div(stride, rounding_mode='floor')
                    valid &= ((co >= 0) & (co < shape_tensor)).all(1)
                    candidates.append(co[valid])
                joined = torch.cat(candidates)
                unique_keys = torch.unique(keys(joined,out_shape), sorted=True)
                x = unique_keys // (out_shape[1]*out_shape[2])
                y = unique_keys // out_shape[2] % out_shape[1]
                z = unique_keys % out_shape[2]
                out_coords = torch.stack((x,y,z),1)
            in_keys = keys(state.coords,in_shape)
            sorted_keys, permutation = in_keys.sort()
            pairs=[]
            in_bounds = torch.tensor(in_shape,device=device)
            for offset in offsets:
                query = out_coords * stride - padding + torch.tensor(offset,device=device)
                valid = ((query >= 0) & (query < in_bounds)).all(1)
                query_keys = keys(query,in_shape)
                index = torch.searchsorted(sorted_keys,query_keys).clamp_max(len(sorted_keys)-1)
                valid &= sorted_keys[index] == query_keys
                out_index = valid.nonzero().flatten()
                in_index = permutation[index[valid]]
                pairs.append((out_index,in_index))
            if self.subm:
                state.cache[cache_key] = (out_coords,out_shape,pairs)
        result = torch.zeros((len(out_coords),self.weight.shape[-1]),device=device,dtype=state.features.dtype)
        for offset,(out_index,in_index) in zip(offsets,pairs):
            result[out_index] += state.features[in_index] @ self.weight[offset]
        return SparseState(result,out_coords,out_shape,state.cache if self.subm else {})


class SparseSequence(nn.Sequential):
    def forward(self,state):
        for module in self:
            if isinstance(module,(ReferenceSparseConv,SparseBlock,SparseSequence)):
                state=module(state)
            else:
                state=state.replace(module(state.features))
        return state


class SparseBlock(nn.Module):
    def __init__(self,channels):
        super().__init__()
        self.conv1=ReferenceSparseConv(channels,channels,3,padding=1,subm=True)
        self.bn1=nn.BatchNorm1d(channels,eps=1e-3,momentum=0.01)
        self.conv2=ReferenceSparseConv(channels,channels,3,padding=1,subm=True)
        self.bn2=nn.BatchNorm1d(channels,eps=1e-3,momentum=0.01)

    def forward(self,state):
        x=self.conv1(state)
        x=x.replace(F.relu(self.bn1(x.features)))
        x=self.conv2(x)
        return x.replace(F.relu(self.bn2(x.features)+state.features))


def convmodule(cin,cout,kernel=3,stride=1,padding=0,subm=False):
    return SparseSequence(ReferenceSparseConv(cin,cout,kernel,stride,padding,subm),
                          nn.BatchNorm1d(cout,eps=1e-3,momentum=0.01), nn.ReLU())


class ReferenceSparseEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_input=convmodule(5,16,padding=1,subm=True)
        self.encoder_layers=SparseSequence()
        for index,channels in enumerate((16,32,64,128)):
            blocks=[SparseBlock(channels),SparseBlock(channels)]
            if index<3:
                blocks.append(convmodule(channels,channels*2,stride=2,padding=(1,1,0) if index==2 else 1))
            self.encoder_layers.add_module(f'encoder_layer{index+1}',SparseSequence(*blocks))
        self.conv_out=convmodule(128,128,kernel=(1,1,3),stride=(1,1,2))

    def forward(self,features,coords,verbose=True):
        if features.shape[0] == 0:
            return features.new_zeros((1, 256, 128, 128))
        state=SparseState(features,coords.long(),(1024,1024,41))
        start=time.perf_counter()
        state=self.conv_input(state)
        if verbose: print('SPARSE input',len(state.coords),flush=True)
        for name,layer in self.encoder_layers.named_children():
            state=layer(state)
            if verbose: print('SPARSE',name,len(state.coords),state.shape,round(time.perf_counter()-start,2),flush=True)
        state=self.conv_out(state)
        flat=keys(state.coords,state.shape)
        dense=state.features.new_zeros((state.shape[0]*state.shape[1]*state.shape[2],128))
        dense[flat]=state.features
        dense=dense.view(*state.shape,128).permute(3,2,0,1).contiguous()
        return dense.reshape(1,256,state.shape[0],state.shape[1])
