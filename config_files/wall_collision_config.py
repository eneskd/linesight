"""
Configuration for wall collision penalty system.

This module provides configuration options for adding punishment when the LSTM agent hits walls.
"""

# Wall collision penalty schedule - negative values for punishment
# Format: (step, penalty_value)
wall_collision_penalty_schedule = [
    (0, 0.0),                    # Start with no penalty
    (500_000, -0.1),            # Small penalty after 500k steps
    (1_000_000, -0.2),          # Moderate penalty after 1M steps
    (2_000_000, -0.5),          # Higher penalty after 2M steps
    (5_000_000, -1.0),          # Strong penalty for advanced training
]

# Alternative: Fixed penalty (uncomment to use instead of schedule)
# FIXED_WALL_COLLISION_PENALTY = -0.5

# Whether to enable wall collision detection and penalty
ENABLE_WALL_COLLISION_PENALTY = True

# Minimum speed threshold for wall collision penalty (to avoid penalizing stationary contact)
MIN_SPEED_FOR_WALL_PENALTY = 1.0  # m/s

# Whether to apply penalty only during forward motion
PENALTY_ONLY_DURING_FORWARD_MOTION = True