"""
LLM-based agent implementing AbstractAgent interface.
Uses LLM for decision making in MiniGrid environments.
"""

from typing import Any, Dict, Tuple

import os
import random
import re
import sys

from agent.abstract_agent import AbstractAgent

from .llm_interface import LLMInterface
from .observation_parser import ObservationParser
from .prompt_manager import PromptManager

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class LLMAgent(AbstractAgent):
    """LLM-based agent for MiniGrid environments."""

    def __init__(
        self, env_name: str, model: str = "gpt-4", debug: bool = False, **kwargs
    ):
        """
        Initialize LLM agent.

        Args:
            env_name: Name of the MiniGrid environment
            model: LLM model to use
            debug: Enable debug printing
            **kwargs: Additional arguments for LLM interface
        """
        super().__init__()

        self.env_name = env_name
        self.model = model
        self.debug = debug

        # Initialize components
        self.llm_interface = LLMInterface(model=model, **kwargs)
        self.observation_parser = ObservationParser()
        self.prompt_manager = PromptManager(env_name=env_name)

        # Agent state
        self.action_space_size = None
        self.last_action = None
        self.is_went_through_door = False

    def predict_action(self, observation: Any, **kwargs) -> Tuple[int, Dict[str, Any]]:
        """Get action from LLM given observation"""
        # Convert observation to text
        state_desc = self.observation_parser.parse_observation(observation)

        # Create prompt
        prompt = self.prompt_manager.create_query_prompt(
            state_desc, self.is_went_through_door
        )
        # Query LLM
        try:
            if self.debug:
                print(f"Querying LLM with prompt: {prompt}")

            response = self.llm_interface.query(
                prompt, system_prompt=self.prompt_manager.get_system_prompt()
            )

            if self.debug:
                print(f"Response:\n {response}")

            # Parse action from response
            action = self._parse_action_from_response(response)

            info = {
                "llm_response": response,
                "state_description": state_desc,
            }

            return action, info

        except Exception as e:
            print(f"LLM query failed: {e}")
            raise e

    def _parse_action_from_response(self, response):
        """Parse action number from LLM response"""
        # Action: number
        numbers = re.findall(r"Action:\s*(\d+)", response)

        if numbers:
            action = int(numbers[0])
            # Validate action
            if self.action_space_size and 0 <= action < self.action_space_size:
                return action
            else:
                print(
                    f"Invalid action {action}, must be 0-{self.action_space_size - 1}"
                )
                return random.randint(0, 1)  # fallback to turn left / right

        print(f"Could not parse action from LLM response: {response}")
        return random.randint(0, 1)  # fallback to turn left / right

    def reset_episode(self):
        """Reset episode-specific state."""
        self.episode_step = 0
        self.is_went_through_door = False
        self.observation_parser.is_open_door = False

    def update_agent(self, **kwargs) -> None:
        """Update agent parameters."""
        if "action_space_size" in kwargs:
            self.action_space_size = kwargs["action_space_size"]

    # Don't need these for LLM agent
    def save(self, **kwargs) -> None:
        pass

    def load(self, **kwargs) -> None:
        pass
