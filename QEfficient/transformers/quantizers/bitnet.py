from typing import List
import torch
from torch import nn
import torch.nn.functional as F
from QEfficient.transformers.quantizers.quantizer_utils import BITNET_VALUES_PER_ITEM, unpack_weights_bitnet



class BitLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int, bias: bool, device=None, dtype=None):
        super().__init__()
        self.dtype = dtype
        self.register_buffer(
            "weight",
            torch.zeros(
                (out_features // BITNET_VALUES_PER_ITEM, in_features),
                dtype=torch.uint8,
                device=device,
            ),
        )
        self.register_buffer(
            "weight_scale",
            torch.ones(
                (1),
                dtype=dtype,
                device=device,
            ),
        )
        if bias:
            self.register_buffer("bias", torch.zeros((out_features), dtype=dtype, device=device))
        else:
            self.bias = None

    @torch.compile
    def activation_quant(self, input, num_bits=8):
        """
        Activation function : Performs symmetric, per-token quantization on the input activations.
        Parameters:
        -----------
        x : torch.Tensor
            Input activations to be quantized.
        num_bits : int, optional (default=8)
            Number of bits to use for quantization, determining the quantization range.

        Returns:
        --------
        result : torch.Tensor
            Quantized activation tensor, with values mapped to an `int8` range.
        scale : torch.Tensor
            The per-channel scaling factors used to quantize the tensor.
        """
        Qn = -(2 ** (num_bits - 1))
        Qp = 2 ** (num_bits - 1) - 1
        scale = Qp / input.abs().max(dim=-1, keepdim=True).values.clamp(min=1e-5)
        result = (input * scale).round().clamp(Qn, Qp)
        return result.to(torch.int8), scale

    @torch.compile
    def post_quant_process(self, input, input_scale, weight_scale):
        out = input / (input_scale * weight_scale)
        return out

    def forward(self, input):
        w = self.weight
        w_quant = unpack_weights_bitnet(w, dtype=self.dtype)
        input_quant, input_scale = self.activation_quant(input)
        y = F.linear(input_quant.to(self.dtype), w_quant)
        y = self.post_quant_process(y, self.weight_scale, input_scale)
        if self.bias is not None:
            y += self.bias.reshape(1, -1).expand_as(y)
        return y

