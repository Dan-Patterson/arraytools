from .arr_moving import block, stride_, rolling_stats
from .compass_angles import compass
from .line_ang_azim import line_dir
from .near import not_closer, n_near
from .vincenty import vincenty
__all__ = ['block',
           'stride_',
           'rolling_stats',
           'compass',
           'line_dir',
           'not_closer',
	       'n_near',
	       'vincenty']
