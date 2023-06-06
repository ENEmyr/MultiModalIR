class AvgMeter(object):
    """
    Keep running average for a metric
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.avg = None
        self.sum = None
        self.cnt = 0

    def update(self, val, n=1):
        if not self.sum:
            self.sum = val * n
        else:
            self.sum += val * n
        self.cnt += n
        self.avg = self.sum / self.cnt
