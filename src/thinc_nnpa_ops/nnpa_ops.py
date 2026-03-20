from thinc_bigendian_ops import BigEndianOps
from thinc.config import registry
from thinc.compat import torch

@registry.ops("nnpa")
class NnpaOps(BigEndianOps):
    """Thinc Ops class that handles usage of Telum acceleration.
    Basic operations fall back to the big-endian numpy implementation.
    This largely enables the use of the nnpa device for pytorch models."""
    name = "nnpa"

    def has_gpu_support(self):
        return torch != None and hasattr(torch, "nnpa")

    def get_default_torch_device(self):
        return torch.device("nnpa")