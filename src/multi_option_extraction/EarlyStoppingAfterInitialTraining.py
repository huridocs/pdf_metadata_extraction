from transformers import EarlyStoppingCallback


class EarlyStoppingAfterInitialTraining(EarlyStoppingCallback):
    def on_evaluate(self, args, state, control, metrics, **kwargs):
        if state.epoch < 18:
            return
        return super().on_evaluate(args, state, control, metrics, **kwargs)