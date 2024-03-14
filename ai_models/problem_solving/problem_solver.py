# ai_models/problem_solving/problem_solver.py

from queue import PriorityQueue
from ai_models.problem_solving.problem_solver_utils import goal_state, get_successors, heuristic

class ProblemSolver:
    def __init__(self):
        pass

    def solve_problem(self, initial_state):
        visited = set()
        priority_queue = PriorityQueue()
        priority_queue.put((0, initial_state, []))

        while not priority_queue.empty():
            _, current_state, path = priority_queue.get()

            if goal_state(current_state):
                return path

            if current_state in visited:
                continue

            visited.add(current_state)

            for successor_state, action, cost in get_successors(current_state):
                if successor_state not in visited:
                    new_cost = len(path) + cost
                    priority = new_cost + heuristic(successor_state, goal_state)
                    priority_queue.put((priority, successor_state, path + [action]))

        return None