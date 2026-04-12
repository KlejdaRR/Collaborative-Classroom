class SimpleTeacher:
    def __init__(self, total_calls=2000):
        self.calls_remaining = total_calls
        self.total_calls = total_calls

    def teach(self, x, y_true):
        if self.calls_remaining > 0:
            self.calls_remaining -= 1
            return y_true
        return None

    def get_remaining(self):
        return self.calls_remaining

    def get_usage(self):
        return 100 * (self.total_calls - self.calls_remaining) / self.total_calls