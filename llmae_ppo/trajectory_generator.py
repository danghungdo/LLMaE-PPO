"""
LLM Trajectory Generator

Generates expert trajectories using LLM agents for behavioral cloning pre-training.
Saves successful trajectories as pickle files containing state-action pairs.
"""

from typing import Any, Dict

import argparse
import json
import os
import pickle
import sys
import time

import gymnasium as gym

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from trajectory_generation import LLMAgent
from utils import set_seed


class TrajectoryCollector:
    """Collects and saves trajectories for behavioral cloning."""

    def __init__(self, args):
        """
        Initialize trajectory collector.

        Args:
            args: Command line arguments
        """
        self.args = args
        self.successful_trajectories = []
        self.failed_trajectories = []
        self.total_generated = 0
        self.success_count = 0

        # Create output directory
        os.makedirs(args.output_dir, exist_ok=True)

    def generate_trajectories(self) -> Dict[str, Any]:
        """
        Generate trajectories using LLM agent.

        Returns:
            Dictionary with generation statistics
        """
        print(f"Generating {self.args.num_trajectories} trajectories...")
        print(f"Environment: {self.args.env_name}")
        print(f"LLM Model: {self.args.model}")
        print("-" * 60)

        start_time = time.time()

        # Generate trajectories with different seeds (1 trajectory per seed)
        seeds = list(
            range(
                self.args.start_seed, self.args.start_seed + self.args.num_trajectories
            )
        )

        for i, seed in enumerate(seeds):
            success = self._generate_single_trajectory(seed, i)

            if success:
                self.success_count += 1

            self.total_generated += 1

            # Print progress
            if (i + 1) % 5 == 0:
                success_rate = self._get_success_rate()
                print(
                    f"Progress: {self.total_generated}/{self.args.num_trajectories} "
                    f"(Success rate: {success_rate:.3f})"
                )

        # Save collected trajectories
        if self.successful_trajectories:
            self._save_metadata()

        # Generate summary
        end_time = time.time()
        duration = end_time - start_time

        summary = {
            "total_generated": self.total_generated,
            "successful": self.success_count,
            "failed": len(self.failed_trajectories),
            "success_rate": self._get_success_rate(),
            "duration_seconds": duration,
            "trajectories_saved": len(self.successful_trajectories),
        }

        self._print_summary(summary)
        return summary

    def _generate_single_trajectory(self, seed: int, trajectory_idx: int) -> bool:
        """Generate a single trajectory."""
        try:
            # Create environment
            render_mode = "human" if self.args.visualize else None
            env = gym.make(self.args.env_name, render_mode=render_mode)

            # Create LLM agent
            agent = LLMAgent(
                env_name=self.args.env_name,
                model=self.args.model,
                debug=self.args.debug,
                max_retries=3,
                retry_delay=1.0,
            )
            agent.update_agent(action_space_size=env.action_space.n)

            # Generate trajectory
            trajectory_data = self._run_episode(env, agent, seed)
            env.close()

            # Check if trajectory was successful
            success = trajectory_data["success"]

            # Only save successful trajectories
            if success:
                trajectory_data["trajectory_id"] = trajectory_idx
                trajectory_data["seed"] = seed
                self.successful_trajectories.append(trajectory_data)

                # Save individual trajectory file
                self._save_single_trajectory(trajectory_data, trajectory_idx, seed)

            return success

        except Exception as e:
            print(f"Failed to generate trajectory {trajectory_idx} (seed {seed}): {e}")
            raise e

    def _run_episode(self, env, agent, seed: int) -> Dict[str, Any]:
        """Run a single episode and collect data."""
        obs, _ = env.reset(seed=seed)
        agent.reset_episode()

        states = []
        actions = []
        rewards = []
        episode_reward = 0
        steps = 0

        done = False
        while not done and steps < self.args.max_steps:
            # Render if visualization is enabled
            if self.args.visualize:
                env.render()

            # Get observation for BC (flattened state)
            if isinstance(obs, dict) and "image" in obs:
                flat_state = obs["image"].flatten()
            else:
                flat_state = obs

            # Get action from agent
            action, info = agent.predict_action(obs)

            # Debug output
            if self.args.debug:
                print(f"\n--- Step {steps} ---")
                print(f"State: {info.get('state_description', 'N/A')}")
                print(f"LLM Response: {info.get('llm_response', 'N/A')}")
                print(f"Extracted Action: {action}")

            # Handle door tracking logic
            if (
                agent.observation_parser.is_open_door
                and action == 2
                and not agent.is_went_through_door
            ):
                if self.args.debug:
                    print("Agent went through the door!")
                agent.is_went_through_door = True

            # Store state-action pair
            states.append(flat_state)
            actions.append(action)

            # Execute action
            obs, reward, terminated, truncated, env_info = env.step(action)

            rewards.append(reward)
            episode_reward += reward
            steps += 1
            done = terminated or truncated

        # Determine if trajectory was successful
        success = terminated and episode_reward > 0

        return {
            "states": states,
            "actions": actions,
            "rewards": rewards,
            "total_reward": episode_reward,
            "steps": steps,
            "success": success,
            "terminated": terminated,
            "truncated": truncated,
        }

    def _save_single_trajectory(self, trajectory_data, trajectory_idx, seed):
        """Save a single trajectory as individual pickle file."""
        # Convert to state-action pairs for BC training
        state_action_pairs = []
        for state, action in zip(trajectory_data["states"], trajectory_data["actions"]):
            state_action_pairs.append({"state": state, "action": action})

        # Save individual trajectory file
        filename = os.path.join(
            self.args.output_dir, f"trajectory_{trajectory_idx}_seed_{seed}.pkl"
        )
        with open(filename, "wb") as f:
            pickle.dump(state_action_pairs, f)

        if self.args.debug:
            print(
                f"Saved trajectory {trajectory_idx} (seed {seed}): {len(state_action_pairs)} state-action pairs to {filename}"
            )

    def _save_metadata(self):
        """Save trajectory metadata as JSON file."""
        metadata = {
            "args": vars(self.args),
            "successful_trajectories": len(self.successful_trajectories),
            "failed_trajectories": len(self.failed_trajectories),
            "success_rate": self._get_success_rate(),
            "total_state_action_pairs": sum(
                len(traj["states"]) for traj in self.successful_trajectories
            ),
            "generation_timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "successful_seeds": [traj["seed"] for traj in self.successful_trajectories],
        }

        filename = os.path.join(self.args.output_dir, "metadata.json")
        with open(filename, "w") as f:
            json.dump(metadata, f, indent=2)
        print(f"Saved metadata to {filename}")

    def _get_success_rate(self) -> float:
        """Calculate current success rate."""
        if self.total_generated == 0:
            return 0.0
        return self.success_count / self.total_generated

    def _print_summary(self, summary: Dict[str, Any]):
        """Print generation summary."""
        print("\n" + "=" * 60)
        print("TRAJECTORY GENERATION SUMMARY")
        print("=" * 60)
        print(f"Total trajectories generated: {summary['total_generated']}")
        print(f"Successful trajectories: {summary['successful']}")
        print(f"Failed trajectories: {summary['failed']}")
        print(f"Success rate: {summary['success_rate']:.3f}")
        print(f"Duration: {summary['duration_seconds']:.1f} seconds")
        print(f"Trajectories saved: {summary['trajectories_saved']}")

        if summary["trajectories_saved"] > 0:
            print(f"Output directory: {self.args.output_dir}")
        print("=" * 60)


def main():
    """Main trajectory generation function."""
    parser = argparse.ArgumentParser(
        description="Generate LLM trajectories for behavioral cloning"
    )
    parser.add_argument(
        "--env_name", default="MiniGrid-DoorKey-8x8-v0", help="Environment name"
    )
    parser.add_argument(
        "--model", default="meta-llama/llama-3.3-70b-instruct", help="LLM model"
    )
    parser.add_argument(
        "--num_trajectories",
        type=int,
        default=50,
        help="Number of trajectories to generate",
    )
    parser.add_argument("--start_seed", type=int, default=42, help="Starting seed")
    parser.add_argument(
        "--max_steps", type=int, default=50, help="Max steps per trajectory"
    )
    parser.add_argument(
        "--output_dir", default="trajectory_data_1", help="Output directory"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode (print observations and LLM responses)",
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Enable visualization (render environment)",
    )

    args = parser.parse_args()

    print("LLM Trajectory Generator")
    print("=" * 60)

    # Set random seed
    set_seed(args.start_seed)

    # Create collector and generate trajectories
    collector = TrajectoryCollector(args)
    _ = collector.generate_trajectories()


if __name__ == "__main__":
    main()
