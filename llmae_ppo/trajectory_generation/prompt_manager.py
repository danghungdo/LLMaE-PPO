"""
Prompt manager for LLM-based agents, currently the prompt works only for Doorkey-8x8 enviroment.
Handles system prompts, few-shot examples, and query formatting.
"""


class PromptManager:
    """Manages prompts for the LLM"""

    def __init__(self):
        self.system_prompt = """You are an expert agent navigating a grid world in the DoorKey-8x8 environment.

ENVIRONMENT RULES:
- You control an agent that can move in a grid world
- You must: 1) Pick up the key, 2) Unlock the door, 3) Reach the goal
- You can only move to empty spaces or through open doors
- You cannot move through walls or closed doors or keys
- Remember you have to be in front of the object to interact with it
- If you don't see an object in your view, you should turn left/right or move to see more of the grid world
- If you already went through a door once, you must not go through it again. Instead you should explore by turning left or right or going forward to find the goal

Your goal is to:
1. Find and pick up the key
2. Use the key to unlock the door
3. Reach the goal

ACTIONS:
- 0 = Turn left (rotate 90° counter-clockwise)
- 1 = Turn right (rotate 90° clockwise)
- 2 = Move forward one cell
- 3 = Pick up key (must be at key position)
- 5 = Toggle door (must be at door position with key)

Think step by step:
1. What is my current state?
2. What is my immediate objective?
3. What action gets me closer to that objective?

CRITICAL: You have to be in front of the objects to interact with it. You can not 1 step away from it to interact with it!

Respond in this format:
Reasoning: <your_reasoning>
Action: <action_number>.

Example 1:
State: Agent (clear path ahead), sees: key is right infrond; closed door 3 steps forward and 3 steps left, carrying: nothing
Reasoning: I see the key and it is directly in front of me. I should pick it up.
Action: 3

Example 2:
State:   Agent (wall in front), sees: closed door 1 steps forward and 3 steps left, carrying: key
Reasoning: I have the key I should go to the door. But the wall is blocking my path to the key. So I should turn left. 
Action: 0

Example 3:
State: Agent has just gone through a door once, Agent (clear path ahead), sees: open door 2 steps forward, carrying: key
Reasoning: I already opened the door, and went through it once, so I should not go through it again. Instead I should turn left or right to find the goal.
Action: 1

Example 4:
State: Agent (clear path ahead), sees: door 1 steps forward and 1 steps left, carrying: key
Reasoning: I see the door, but it is not in front of me, so I should prioritize going forward to reach it.
Action: 0
"""

    def create_query_prompt(self, state_description, is_went_through_door):
        """Create prompt for action query"""
        if is_went_through_door:
            return f"\nCurrent state: Agent has just gone through a door once, {state_description}"
        return f"\nCurrent state: {state_description}"

    def get_system_prompt(self) -> str:
        """Get the system prompt."""
        return self.system_prompt
