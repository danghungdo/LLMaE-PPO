"""
Observation parser for MiniGrid environments.
Converts grid observations to natural language descriptions for LLM consumption.
"""


class ObservationParser:
    """Parses MiniGrid observations into natural language descriptions."""

    def __init__(self):
        self.OBJECT_TO_IDX = {
            "unseen": 0,
            "empty": 1,
            "wall": 2,
            "floor": 3,
            "door": 4,
            "key": 5,
            "ball": 6,
            "box": 7,
            "goal": 8,
            "lava": 9,
            "agent": 10,
        }

        self.IDX_TO_OBJECT = {v: k for k, v in self.OBJECT_TO_IDX.items()}

        self.DIR_TO_STR = {0: "right", 1: "down", 2: "left", 3: "up"}

        self.is_open_door = False

    def parse_observation(self, obs) -> str:
        """Parse the observation to detailed text with spatial info."""
        if isinstance(obs, dict) and "image" in obs:
            image = obs["image"]
        else:
            image = obs

        h, w, _ = image.shape

        agent_pos = (w // 2, h - 1)  # Agent is at the bottom center of the view

        # get observation channels
        obj_grid = image[:, :, 0]  # Object types
        state_grid = image[:, :, 2]  # State types

        # get agent direction
        agent_dir_idx = obs["direction"]

        return self._build_spatial_description(
            obj_grid, state_grid, agent_pos, agent_dir_idx
        )

    def _build_spatial_description(
        self, obj_grid, state_grid, agent_pos, agent_dir_idx
    ):
        """Build a detailed spatial description of the observation."""

        object_info = []
        h, w = obj_grid.shape

        for i in range(h):
            for j in range(w):
                obj_idx = obj_grid[i, j]

                # skip unseen, empty, wall
                if obj_idx in [0, 1]:
                    continue
                # skip agent position
                if agent_pos == (i, j):
                    continue
                obj_type = self.IDX_TO_OBJECT.get(obj_idx)

                if obj_type in ["key", "door", "goal"]:
                    state = state_grid[i, j]

                    # Get relative position from agent's perspective
                    relative_pos = self._get_relative_position(
                        agent_pos, (i, j), agent_dir_idx
                    )

                    info = {
                        "type": obj_type,
                        "relative_position": relative_pos,
                        "distance": abs(i - agent_pos[0]) + abs(j - agent_pos[1]),
                    }

                    if obj_type == "door":
                        info["is_open"] = state == 0

                    object_info.append(info)

        carrying = "key" if obj_grid[agent_pos] == 5 else "nothing"
        # carrying = 'nothing'

        front = (agent_pos[0], agent_pos[1] - 1)

        is_wall_in_front = obj_grid[front] == self.OBJECT_TO_IDX["wall"]

        return self._format_description(object_info, carrying, is_wall_in_front)

    def _get_relative_position(self, agent_pos, obj_pos, agent_dir):
        """Get object position relative to agent's facing direction"""

        # print(f"Agent pos: {agent_pos}, Object pos: {obj_pos}, Agent dir: {agent_dir}")

        # Calculate raw offset
        dy = agent_pos[1] - obj_pos[1]  # alway postive or equal to 0
        dx = obj_pos[0] - agent_pos[0]  # postive is right, negative is left

        # Convert to description
        position_parts = []

        if dx == 0 and dy == 1:
            return "is right infront"

        if dy > 0:  # Object is to the right
            position_parts.append(f"{dy} steps forward")
        if dx > 0:
            position_parts.append(f"{dx} steps right")
        if dx < 0:  # Object is to the left
            position_parts.append(f"{-dx} steps left")

        if not position_parts:
            return "at agent position"

        return " and ".join(position_parts)

    def _format_description(self, objects_info, carrying, is_wall_in_front):
        """Format the complete description"""

        parts = []
        if is_wall_in_front:
            parts.append("Agent (wall in front)")
        else:
            parts.append("Agent (clear path ahead)")

        # Objects in view
        if objects_info:
            obj_descriptions = []

            # Sort by distance for consistent ordering
            objects_info.sort(key=lambda x: x["distance"])

            for obj in objects_info:
                if obj["type"] == "door":
                    state = "open" if obj["is_open"] else "closed"
                    if state == "open":
                        self.is_open_door = True
                    desc = f"{state} door {obj['relative_position']}"
                else:
                    desc = f"{obj['type']} {obj['relative_position']}"
                obj_descriptions.append(desc)

            parts.append(f"sees: {'; '.join(obj_descriptions)}")
        else:
            parts.append("sees: nothing")

        # Carrying status
        parts.append(f"carrying: {carrying}")

        return ", ".join(parts)
