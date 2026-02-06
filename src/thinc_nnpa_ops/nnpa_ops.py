from thinc_bigendian_ops import BigEndianOps
from thinc.config import registry

@registry.ops("NnpaOps")
class NnpaOps(BigEndianOps):
    """Thinc Ops class that handles usage of Telum acceleration.
    Basic operations fall back to the big-endian numpy implementation.
    This largely enables the use of the nnpa device for pytorch models."""
    name = "nnpa"
