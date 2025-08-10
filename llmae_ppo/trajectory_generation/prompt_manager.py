"""
Prompt manager for LLM-based agents.
Handles system prompts, few-shot examples, and query formatting.
"""


class PromptManager:
    """Manages prompts and templates for LLM interaction."""

    def __init__(self, env_name: str = "MiniGrid-DoorKey-8x8-v0"):
        """
        Initialize prompt manager for specific environment.

        Args:
            env_name: Name of the MiniGrid environment
        """
        self.env_name = env_name
        self.system_prompt = self._get_system_prompt()
        self.few_shot_examples = self._get_few_shot_examples()

    def _get_system_prompt(self) -> str:
        """Get system prompt based on environment."""
        if "DoorKey" in self.env_name:
            return self._get_doorkey_system_prompt()
        else:
            return self._get_generic_system_prompt()

    def _get_doorkey_system_prompt(self) -> str:
        """System prompt for DoorKey environments."""
        return """You are an expert agent navigating a grid world in the DoorKey environment.

ENVIRONMENT RULES:
- You control an agent that can move in a grid world
- You must: 1) Pick up the key, 2) Unlock the door, 3) Reach the goal
- You can only move to empty spaces or through open doors
- You cannot move through walls or closed doors
- You must be directly in front of objects to interact with them
- If you don't see an object, explore by turning left/right or moving forward

OBJECTIVE:
1. Find and pick up the key
2. Use the key to unlock the door  
3. Navigate to the goal

ACTIONS:
- 0 = Turn left (rotate 90° counter-clockwise)
- 1 = Turn right (rotate 90° clockwise)
- 2 = Move forward one cell
- 3 = Pick up key (must be directly in front of key)
- 5 = Toggle door (must be directly in front of door with key)

STRATEGY:
1. Analyze your current state
2. Identify your immediate objective
3. Choose the action that gets you closer to that objective
4. Be precise about positioning - you must be directly in front of objects to interact

CRITICAL: You must be directly in front of objects to interact with them!

Respond in this format:
Reasoning: <your_reasoning>
Action: <action_number>"""

    def _get_generic_system_prompt(self) -> str:
        """Generic system prompt for other environments."""
        return """You are an expert agent navigating a MiniGrid environment.

ACTIONS:
- 0 = Turn left
- 1 = Turn right  
- 2 = Move forward
- 3 = Pick up object
- 5 = Toggle/interact with object

Analyze the situation and choose the best action to reach your goal.

Respond in this format:
Reasoning: <your_reasoning>
Action: <action_number>"""

    def _get_few_shot_examples(self) -> str:
        """Get few-shot examples for the environment."""
        if "DoorKey" in self.env_name:
            return """
Example 1:
State: Agent (clear path ahead), sees: key is right in front, carrying: nothing
Reasoning: I see the key directly in front of me. I should pick it up.
Action: 3

Example 2:  
State: Agent (wall in front), sees: closed door 1 steps forward and 3 steps left, carrying: key
Reasoning: There's a wall blocking my path. I need to turn to navigate around it.
Action: 0

Example 3:
State: Agent (clear path ahead), sees: closed door is right in front, carrying: key  
Reasoning: I have the key and I'm directly in front of the closed door. I should unlock it.
Action: 5

Example 4:
State: Agent (clear path ahead), sees: goal 2 steps forward, carrying: key
Reasoning: I can see the goal ahead. I should move forward to reach it.
Action: 2
"""
        else:
            return ""

    def create_query_prompt(self, state_description, is_went_through_door):
        """Create prompt for action query"""
        # prompt_parts = []
        # prompt_parts.append(f"\nCurrent state: {state_description}")
        if is_went_through_door:
            return f"\nCurrent state: Agent has just gone through a door once, {state_description}"
        return f"\nCurrent state: {state_description}"

    def get_system_prompt(self) -> str:
        """Get the system prompt."""
        return self.system_prompt
