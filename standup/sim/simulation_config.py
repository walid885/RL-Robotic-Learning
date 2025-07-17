# simulation/simulation_config.py
class SimulationConfig:
    def __init__(self):
        self.time_step = 1.0 / 240.0 # Physics simulation time step
        self.simulation_rate = 240 # GUI rendering rate (steps per second)
        self.stabilization_steps = 240 # Steps to run after reset for stabilization
        self.robot_height = 1.0 # Initial spawn height for the robot's base
        
        # Dynamics parameters
        self.linear_damping = 0.01
        self.angular_damping = 0.01
        self.max_joint_velocity = 10.0 # For changeDynamics

        # RL specific parameters
        self.sim_steps_per_action = 10 # Number of physics steps for each RL action
        self.target_standing_height = 0.8 # Target Z-height for reward
        self.min_stable_height = 0.5 # Min Z-height before terminating episode
        self.max_tilt_angle_cos = 0.7 # Cosine of max angle for uprightness (e.g., cos(45 deg) is ~0.707)

        # Reward weights (tune these!)
        self.reward_weight_height = 1.0
        self.reward_weight_uprightness = 2.0
        self.penalty_weight_horizontal_velocity = -0.1
        self.penalty_weight_action_magnitude = -0.01
        self.penalty_weight_non_foot_contact = -5.0
        self.penalty_episode_termination = -10.0

        # Joint control parameters
        self.default_joint_force = 500 # Max force for position control