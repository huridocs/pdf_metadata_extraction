from transformers import EarlyStoppingCallback


class EarlyStoppingAfterInitialTraining(EarlyStoppingCallback):
    def on_evaluate(self, args, state, control, model, metrics, **kwargs):
        if state.global_step < 1800:
            return False
        return super().on_evaluate(args, state, control, model, metrics, **kwargs)