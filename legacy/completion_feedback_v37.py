"""Score feedback with native or postprocessed gates; both pay cleanup cost."""
from runtime import CompletionCache as NativeCompletion
from nam_postprocess_v33 import Postprocessor

class FeedbackCompletion(NativeCompletion):
    def __init__(self,capacity=256,cost_mode='clean'):
        super().__init__(capacity)
        assert cost_mode in ['native','clean'];self.cost_mode=cost_mode
        self.processor=Postprocessor(capacity)
        self.last_native=self.last_clean=None;self.last_postprocess_hit=False
    def evaluate(self,graph):
        key,native,hit=super().evaluate(graph)
        clean,pp_hit=self.processor.evaluate(native)
        self.last_native=native;self.last_clean=clean;self.last_postprocess_hit=pp_hit
        return key,(clean if self.cost_mode=='clean'else native),bool(hit and pp_hit)
