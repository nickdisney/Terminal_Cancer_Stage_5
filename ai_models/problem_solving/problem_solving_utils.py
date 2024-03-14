# ai_models/problem_solving/problem_solver_utils.py

def goal_state(state):
    # Define the conditions for the goal state
    # Return True if the state satisfies the goal conditions, False otherwise
    # Example: Check if the user's task is completed successfully
    return state["task_completed"] == True


def get_successors(state):
    # Generate the successor states based on the current state
    # Return a list of tuples (successor_state, action, cost)
    successors = []

    # Example: Generate successor states based on possible actions
    if state["current_action"] == "get_weather":
        # Generate successor states for getting weather information
        successors.append(({**state, "location": "New York"}, "get_weather", 1))
        successors.append(({**state, "location": "London"}, "get_weather", 1))
        # Add more successor states as needed
    elif state["current_action"] == "tell_joke":
        # Generate successor states for telling a joke
        successors.append(({**state, "joke_type": "pun"}, "tell_joke", 1))
        successors.append(({**state, "joke_type": "one-liner"}, "tell_joke", 1))
        # Add more successor states as needed
    # Add more conditions for other actions

    return successors


def heuristic(state, goal_state):
    # Estimate the cost from the current state to the goal state
    # Return a non-negative value representing the estimated cost
    # Example: Use the number of remaining tasks as the heuristic
    remaining_tasks = len(state["tasks"]) - state["completed_tasks"]
    return remaining_tasks