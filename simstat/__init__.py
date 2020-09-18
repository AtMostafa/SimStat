from .permTest import * #since __all__ is defined in each file, this is ok!
from .bootstrap import *
from .bootstrapTest import *

__all__ = (permTest.__all__+bootstrap.__all__+bootstrapTest.__all__)