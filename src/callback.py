from fastcore.all import *
from fastai.callback.core import Callback

class Callback_m(Callback):
    "Temporary that supports custom events"
    def __call__(self, event_name):
        """
        Call `self.{event_name}` without checking that it's defined.
        This temporary solution to allow custom events.
        """
        #_run = (event_name not in _inner_loop or (self.run_train and getattr(self, 'training', True)) or
        #       (self.run_valid and not getattr(self, 'training', False)))
        _run = True
        res = None
        if self.run and _run: res = getattr(self, event_name, noop)()
        if event_name=='after_fit': self.run=True #Reset self.run to True at each end of fit
        return res
