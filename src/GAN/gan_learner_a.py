import torch
from contextlib import contextmanager
from fastai.callback.core import Callback, CancelStepException
from fastcore.all import *
from .abstract_gan_learner import AbstractGANLearner

class GANLearnerA(AbstractGANLearner):
    "This learner is similar to the current GANLearner, but does not support switchers yet."
    def __init__(self, dls, gen, dis, loss, join=False, cbs=None):
        AbstractGANLearner.__init__(self,dls, gen, dis, join, cbs)
        gen_loss_fn = loss.gen_loss_fn()
        dis_loss_fn = loss.dis_loss_fn()
        store_attr('gen_loss_fn,dis_loss_fn')
    
    def create_opt(self):
        self.opt_gen = self.opt_func(self.splitter(self.generator), lr=self.lr)
        self.opt_dis = self.opt_func(self.splitter(self.discriminator), lr=self.lr)
    
    def _do_one_batch_dis(self):
        # Train discriminator
        gb = self.gen_batch(self.b)
        with eval_mode(self.generator), torch.no_grad():
            self.gen_out = self.generate(gb)
        self.db_r = self.dis_batch(self.b)
        self.db_f = self.dis_batch(self.b, self.gen_out)
        self.r_pred, self.f_pred = self.discriminate_x2(self.db_r, self.db_f)
        self.loss_grad = self.dis_loss_fn(detuplify(self.r_pred), detuplify(self.f_pred))
        self.dis_loss = self.loss_grad.clone()
        self('after_dis_loss')
        #self('after_loss')
        # TODO uncomment when TrainEvalCallback or some alternative will be ready to work with this learner
        #if not self.training: return
        self('before_dis_backward')
        #self('before_backward')
        self.loss_grad.backward()
        self._with_events(self.opt_dis.step, 'step', CancelStepException)
        self.opt_dis.zero_grad()
    
    def _do_one_batch_gen(self):
        # Train generator
        gb = self.gen_batch(self.b)
        self.gen_out = self.generate(gb)
        self.db_f = self.dis_batch(self.b, self.gen_out)
        with eval_mode(self.discriminator):
            self.f_pred = self.discriminate(self.db_f)
        self.loss_grad = self.gen_loss_fn(detuplify(self.f_pred))
        self.gen_loss = self.loss_grad.clone()
        self('after_gen_loss')
        #self('after_loss')
        # TODO uncomment when TrainEvalCallback or some alternative will be ready to work with this learner
        #if not self.training: return
        self('before_gen_backward')
        #self('before_backward')
        self.loss_grad.backward()
        self._with_events(self.opt_gen.step, 'step', CancelStepException)
        self.opt_gen.zero_grad()
    
    def _do_one_batch(self):
        # WARNING possibly some cleanup is needed. If we ran only _do_one_batch_gen then
        # self.db_r, self.r_pred from previous step will exist, but will not be used or rewritten at that step.
        # TODO think about memory usage efficiency
        # TODO Here switcher should decide what to do this time
        self._do_one_batch_dis()
        self._do_one_batch_gen()        
    
    def _true_end_cleanup(self):
        self.dis_loss,self.gen_loss = None,None
        self.gb,self.gen_out,self.db_r,self.db_f = (None,),(None,),(None,),(None,)
        self.r_pred,self.f_pred = (None,),(None,)
    
    def _set_opt_hypers(self, lr):
        self.opt_dis.set_hypers(lr=lr)
        self.opt_gen.set_hypers(lr=lr)

@contextmanager
def eval_mode(model):
    try: yield model.eval()
    finally: model.train()

class TempRecorderA(Callback):
    def __init__(self, every_k=1):
        Callback.__init__(self)
        self.step,self.every_k = 0,every_k
        
    def after_batch(self):
        if self.step == self.every_k-1:
            print('gen_loss={:.5f}, dis_loss={:.5f}'.format(self.gen_loss.item(), self.dis_loss.item()))
        self.step = (self.step+1)%self.every_k
