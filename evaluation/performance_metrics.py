class PerformanceMetrics:
    def __init__(self):
        self.metrics = {}

    def add_metric(self, metric_name, metric_func):
        self.metrics[metric_name] = metric_func

    def evaluate_performance(self, data):
        results = {}
        for metric_name, metric_func in self.metrics.items():
            results[metric_name] = metric_func(data)
        return results