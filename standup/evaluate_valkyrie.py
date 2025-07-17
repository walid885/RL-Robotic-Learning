# evaluate_valkyrie.py
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
import os

# Import your custom environment class
from valkyrie_env import ValkyrieStandEnv 

# --- Evaluation Configuration ---
MODEL_PATH = "./valkyrie_standing_model.zip" # Path to your trained model
N_EVAL_EPISODES = 10 # Number of episodes to evaluate
RENDER_MODE = 'human' # 'human' to visualize, 'direct' for headless evaluation

if __name__ == '__main__':
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Trained model not found at {MODEL_PATH}. Please train first.")
    else:
        print(f"Loading trained model from {MODEL_PATH}...")
        # Create the environment for evaluation (single environment for rendering)
        eval_env = ValkyrieStandEnv(render_mode=RENDER_MODE)

        # Load the trained model
        model = PPO.load(MODEL_PATH, env=eval_env)

        print(f"Evaluating policy for {N_EVAL_EPISODES} episodes...")
        # Evaluate the policy
        # `render` in evaluate_policy is for internal SB3 rendering, 
        # but our env already handles it based on its `render_mode`.
        # So setting it to False here to avoid double-rendering conflicts.
        mean_reward, std_reward = evaluate_policy(model, eval_env,
                                                  n_eval_episodes=N_EVAL_EPISODES,
                                                  render=False, 
                                                  return_episode_rewards=False)

        print(f"\nMean reward over {N_EVAL_EPISODES} episodes: {mean_reward:.2f} +/- {std_reward:.2f}")

        # Optional: Run one more episode with human rendering to observe
        print("\nRunning one more episode to visualize performance...")
        obs, _ = eval_env.reset()
        for _ in range(500): # Run for 500 steps (adjust as needed)
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = eval_env.step(action)
            if terminated or truncated:
                print("Visualization episode ended.")
                obs, _ = eval_env.reset()
        eval_env.close()