from fastai.learner import Learner, defaults
from fastcore.all import *

class AbstractLearner(Learner):
    """
    Temporary parent class for all learners specified for any completely different kind of tasks.
    It is builded on top of Learner now yet
    """
    def __init__(self, dls, model, cbs=None):
        Learner.__init__(self,dls,model=None,loss_func=-1,cbs=cbs)
        self.remove_cbs(L(defaults.callbacks))
    
    def _call_one(self, event_name):
        # if not hasattr(event, event_name): raise Exception(f'missing {event_name}')
        for cb in self.cbs.sorted('order'): cb(event_name)
    
    def create_opt(self):
        raise NotImplementedError
    
    def _do_one_batch(self):
        raise NotImplementedError
    
    def _set_device(self, b):
        raise NotImplementedError    

    # TODO может надо разные даталоадеры использовать
    def one_batch(self, i, b):
        self.iter = i
        #b = self._set_device(b)
        #self._split(b)
        self.b = self._set_device(b)
        from fastai.callback.core import CancelBatchException
        self._with_events(self._do_one_batch, 'batch', CancelBatchException)
    
    def _end_cleanup(self):
        #self.dl,self.xb,self.yb,self.pred,self.loss = None,(None,),(None,),None,None
        self.dl,self.b = None,(None,)
        self._true_end_cleanup()
    
    def _true_end_cleanup(self):
        raise NotImplementedError
    
    def _set_opt_hypers(self):
        raise NotImplementedError
    
    def fit(self, n_epoch, lr=None, wd=None, cbs=None, reset_opt=False):
        with self.added_cbs(cbs):
            if reset_opt or not self.opt: self.create_opt()
            if wd is None: wd = self.wd
            #if wd is not None: self.opt.set_hypers(wd=wd)
            self._set_opt_hypers(lr=self.lr if lr is None else lr)
            self.n_epoch = n_epoch
            from fastai.callback.core import CancelFitException
            self._with_events(self._do_fit, 'fit', CancelFitException, self._end_cleanup)

    '''
    def show_training_loop(self):
        raise NotImplementedError
    '''
