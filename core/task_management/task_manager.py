from queue import PriorityQueue

class Task:
    def __init__(self, name, priority, dependencies, executor):
        self.name = name
        self.priority = priority
        self.dependencies = dependencies
        self.executor = executor

    def execute(self):
        self.executor()

class TaskManager:
    def __init__(self):
        self.tasks = PriorityQueue()

    def add_task(self, task):
        self.tasks.put((-task.priority, task))

    def execute_tasks(self):
        while not self.tasks.empty():
            _, task = self.tasks.get()
            for dependency in task.dependencies:
                dependency.execute()
            task.execute()