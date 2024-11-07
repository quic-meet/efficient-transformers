from typing import List
from transformers.quantizers.quantizer_bitnet import BitNetHfQuantizer
from transformers.utils.quantization_config import BitNetConfig
from transformers.utils.import_utils import is_accelerate_available
from QEfficient.transformers.quantizers.bitnet import BitLinear
from QEfficient.transformers.quantizers.quantizer_utils import replace_with_bitnet_linear, get_keys_to_not_convert


class QEffBitNetConfig(BitNetConfig):
    """
    Configuration class for QEffBitNet, extending BitNetConfig.
    This class includes a post-initialization safety checker to ensure that the configuration arguments are correct.
    """


class QEffBitNetQuantizer(BitNetHfQuantizer):
    """
    Quantizer class for QEffBitNet, extending BitNetHfQuantizer.
    This class handles the initialization, environment validation, dtype updating, and model processing for quantization.
    """
    target_cls = BitLinear
    
    def __init__(self, quantization_config: BitNetConfig, **kwargs):
        super().__init__(quantization_config, **kwargs)

    
    def validate_environment(self, *args, **kwargs):
        if not is_accelerate_available():
            raise ImportError("Loading a BitNet quantized model requires accelerate (`pip install accelerate`)")

        if kwargs.get("from_tf", False) or kwargs.get("from_flax", False):
            raise ValueError(
                "Loading ternary weights from tf/flax is currently not supported, please make"
                " sure the weights are in PyTorch format."
            ) 
    
    def _process_model_before_weight_loading(
        self,
        model=None,
        keep_in_fp32_modules: List[str] = [],
        **kwargs,
    ):
        self.modules_to_not_convert = get_keys_to_not_convert(model)

        if self.quantization_config.modules_to_not_convert is not None:
            self.modules_to_not_convert.extend(self.quantization_config.modules_to_not_convert)

        model = replace_with_bitnet_linear(
            model,
            target_cls=self.target_cls,
            modules_to_not_convert=self.modules_to_not_convert,
            quantization_config=self.quantization_config,
            pre_quantized=self.pre_quantized,
        )
        