# mujoco_env package
# This file makes the mujoco_env directory a Python package

# Import necessary modules and classes
try:
    from .mujoco_parser import MuJoCoParserClass
    from .y_env import SimpleEnv
    from .utils import (
        trim_scale,
        compute_view_params,
        get_idxs,
        get_colors,
        get_monitor_size,
        TicTocClass,
    )
    from .transforms import quat2euler, euler2quat
except ImportError:
    # Keep silent if there are import errors
    # This allows the module to be imported even if some dependencies are missing
    pass

# Expose the important classes and functions
__all__ = ['MuJoCoParserClass', 'SimpleEnv']