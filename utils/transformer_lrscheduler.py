class TransformerLrScheduler:
    """
    Obtained from : https://github.com/jreremy/conformer/blob/1c8de9d4bae992c999c47a5dedeedd3feca0f64a/utils.py
    Transformer LR scheduler from "Attention is all you need." https://arxiv.org/abs/1706.03762
    multiplier and warmup_steps taken from conformer paper: https://arxiv.org/abs/2005.08100
    """

    def __init__(self, optimizer, d_model, warmup_steps, multiplier=5):
        self._optimizer = optimizer
        self.d_model = d_model
        self.warmup_steps = warmup_steps
        self.n_steps = 0
        self.multiplier = multiplier

    def step(self):
        self.n_steps += 1
        lr = self._get_lr()
        for param_group in self._optimizer.param_groups:
            param_group["lr"] = lr

    def _get_lr(self):
        return (
            self.multiplier
            * (self.d_model**-0.5)
            * min(self.n_steps ** (-0.5), self.n_steps * (self.warmup_steps ** (-1.5)))
        )
