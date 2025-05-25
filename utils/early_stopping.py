class EarlyStopping:
    def __init__(self, patience=3, mode='max', min_delta=0.0):
        self.patience = patience
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.min_delta = min_delta
        self.mode = mode

    def __call__(self, current_score):
        if self.best_score is None:
            self.best_score = current_score
            return False

        if self._is_improvement(current_score):
            self.best_score = current_score
            self.counter = 0
        else:
            self.counter += 1
            print(f"No improvement. EarlyStopping counter: {self.counter}/{self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
                return True

        return False

    def _is_improvement(self, current):
        if self.mode == 'max':
            return current > self.best_score + self.min_delta
        elif self.mode == 'min':
            return current < self.best_score - self.min_delta
        else:
            raise ValueError("mode must be 'min' or 'max'")
