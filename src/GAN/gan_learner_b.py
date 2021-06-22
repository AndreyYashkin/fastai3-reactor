from fastai.callback.core import Callback
from fastcore.all import *
from .abstract_gan_learner import AbstractGANLearner
from CGDs import ACGD

class GANLearnerB(AbstractGANLearner):
    """
    This GANLearner uses one optimizer designed for minimax problems
    """
    def __init__(self, dls, gen, dis, loss, lr=1e-4, join=False, minimax_opt=ACGD, cbs=None):
        AbstractGANLearner.__init__(self,dls, gen, dis, join, cbs)
        self.minimax_opt = minimax_opt
        loss_fn = loss.loss_fn()
        store_attr('loss_fn')
    
    def create_opt(self):
        #self.lr = 1e-3
        self.opt = self.minimax_opt(min_params=self.splitter(self.generator), max_params=self.splitter(self.discriminator),
                                    lr_max=1e-3, lr_min=1e-3)
    
    def _do_one_batch(self):
        gb = self.gen_batch(self.b)
        self.gen_out = self.generate(gb)
        self.db_r = self.dis_batch(self.b)
        self.db_f = self.dis_batch(self.b, self.gen_out)
        self.r_pred, self.f_pred = self.discriminate_x2(self.db_r, self.db_f)
        self.loss_grad = self.loss_fn(detuplify(self.r_pred), detuplify(self.f_pred))
        self.loss = self.loss_grad.clone()
        self('after_dis_loss')
        #self('after_loss')
        # TODO uncomment when TrainEvalCallback or some alternative will be ready to work with this learner
        #if not self.training: return
        self('before_step') # TODO Another name?
        #self.loss_grad.backward()
        #self._with_events(self.opt, 'step', CancelStepException)
        self.opt.step(loss=self.loss_grad) # TODO Warp optimizer
        #self.opt_dis.zero_grad()
    
    def _true_end_cleanup(self):
        self.loss = None
        self.gen_out,self.db_r,self.db_f = (None,),(None,),(None,)
        self.r_pred,self.f_pred = (None,),(None,)
    
    def _set_opt_hypers(self, lr):
        pass
        #raise NotImplementedError

class TempRecorderB(Callback):
    def __init__(self, every_k=1):
        Callback.__init__(self)
        self.step,self.every_k = 0,every_k
        
    def after_batch(self):
        if self.step == self.every_k-1:
            print('loss={:.5f}'.format(self.loss.item()))
        self.step = (self.step+1)%self.every_k
