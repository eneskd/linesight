"""
Utility functions for reward shaping.
"""

import numpy as np
from numba import jit


# largely inspired from https://github.com/TomashuTTTT7/TM-AlgoCrack/blob/main/cracks/speedslide_quality.py, yet also largely simplified
@jit(nopython=True)
def speedslide_quality_tarmac(speed_x: float, speed_z: float) -> float:
    """
    Extract from Tomashu's documentation:
    - speedslide_quality < 1: you don't utilize entire speedslide potential, steer more.
    - speedslide_quality == 1: you utilize entire speedslide potential.perfect speedslide.
    - speedslide_quality > 1: you utilize entire speedslide potential, but you start losing some speed from drifting, steer less.
    """
    material_max_side_friction_multiplier = 1.0  # will need to be changed in the future for dirt & grass
    max_side_friction = (
        np.interp(speed_z * 3.6, [0, 100, 200, 300, 400, 500], [80, 80, 75, 67, 60, 55]) * material_max_side_friction_multiplier
    )
    side_friction = 20 * abs(speed_x)
    speedslide_quality = (side_friction - max_side_friction) / max_side_friction if side_friction > max_side_friction else 0.0
    return speedslide_quality


@jit(nopython=True)
def speedslide_quality_ratio(lateral_speed: float, forward_speed: float) -> float:
    """
    Calculate the quality of a speedslide based on the ratio of lateral to forward speed.
    
    Args:
        lateral_speed: The lateral velocity of the vehicle (speed_x)
        forward_speed: The forward velocity of the vehicle (speed_z)
        
    Returns:
        A quality score for the speedslide maneuver (0.0 to 1.0)
    """
    if forward_speed == 0:
        return 0.0
    
    # Calculate the ratio of lateral to forward speed
    ratio = abs(lateral_speed / forward_speed)
    
    # Optimal speedslide has a ratio around 1.0
    return 1.0 - abs(ratio - 1.0)


@jit(nopython=True)
def neoslide_quality(lateral_speed: float, forward_speed: float) -> float:
    """
    Calculate the quality of a neoslide maneuver.
    
    Args:
        lateral_speed: The lateral velocity of the vehicle (speed_x)
        forward_speed: The forward velocity of the vehicle (speed_z)
        
    Returns:
        A quality score for the neoslide maneuver
    """
    # A neoslide requires significant lateral speed (>= 2.0)
    if abs(lateral_speed) < 2.0:
        return 0.0
    
    return abs(lateral_speed) / max(1.0, forward_speed)
