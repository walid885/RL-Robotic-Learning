# train_valkyrie.py
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold
from stable_baselines3.common.vec_env import SubprocVecEnv # Required for N_ENVS > 1
import os

# Import your custom environment class
from valkyrie_env import ValkyrieStandEnv 

# --- Training Configuration ---
LOG_DIR = "./valkyrie_standing_logs/"
MODEL_SAVE_PATH = "./valkyrie_standing_model.zip"
TOTAL_TIMESTEPS = 10_000_000 # You'll likely need millions for humanoid standing
EVAL_FREQ = 100_000 # Evaluate every X timesteps
N_ENVS = 4 # Number of parallel environments for faster data collection
RENDER_MODE = 'human' # Set to 'human' to visualize training, 'direct' for headless

# Create log directory if it doesn't exist
os.makedirs(LOG_DIR, exist_ok=True)

if __name__ == '__main__':
    print(f"Starting training for Valkyrie standing. Logs will be saved to: {LOG_DIR}")

    # 1. Create the vectorized environment
    # Use SubprocVecEnv for parallel execution when N_ENVS > 1
    if N_ENVS > 1:
        # Each environment instance will run in its own process.
        # Pass 'direct' as render_mode to the environment if N_ENVS > 1, as only one GUI can exist.
        env_fn = lambda: ValkyrieStandEnv(render_mode='direct')
        vec_env = make_vec_env(env_fn, n_envs=N_ENVS, seed=0, vec_env_cls=SubprocVecEnv)
    else:
        # For single environment, allow GUI rendering if specified
        vec_env = make_vec_env(ValkyrieStandEnv, n_envs=1, seed=0,
                               env_kwargs={'render_mode': RENDER_MODE})
    
    vec_env = Monitor(vec_env, LOG_DIR) # Monitor allows logging rewards

    # 2. Define the PPO model
    # MlpPolicy for a feed-forward neural network policy
    model = PPO("MlpPolicy", vec_env, verbose=1,
                tensorboard_log=LOG_DIR,
                device="cuda" if True else "cpu", # Use "cuda" if you have a GPU, otherwise "cpu"
                n_steps=2048, # Number of steps to run in each environment per policy update
                batch_size=64, # Batch size for policy and value function updates
                gamma=0.99, # Discount factor
                gae_lambda=0.95, # GAE parameter
                n_epochs=10, # Number of epochs to optimize the surrogate loss
                ent_coef=0.01, # Entropy coefficient for exploration
                vf_coef=0.5, # Value function coefficient
                max_grad_norm=0.5 # Max gradient norm for clipping
               )

    # 3. Callbacks for evaluation and saving
    eval_callback = EvalCallback(vec_env,
                                 best_model_save_path=os.path.join(LOG_DIR, "best_model"),
                                 log_path=LOG_DIR,
                                 eval_freq=EVAL_FREQ,
                                 deterministic=True,
                                 render=False) 

    print(f"Training PPO for {TOTAL_TIMESTEPS} timesteps...")
    # 4. Train the model
    model.learn(total_timesteps=TOTAL_TIMESTEPS,
                callback=eval_callback,
                progress_bar=True)

    # 5. Save the final model
    model.save(MODEL_SAVE_PATH)
    print(f"Training finished. Model saved to {MODEL_SAVE_PATH}")

    vec_env.close()