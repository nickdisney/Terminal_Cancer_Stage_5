class SafetyMonitor:
    def __init__(self):
        self.safety_rules = []

    def add_safety_rule(self, rule_func):
        self.safety_rules.append(rule_func)

    def check_safety(self, action):
        for rule_func in self.safety_rules:
            if not rule_func(action):
                return False
        return True