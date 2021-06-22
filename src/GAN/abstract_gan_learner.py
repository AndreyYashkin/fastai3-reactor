import torch
from fastcore.all import *
from fastai.torch_core import *
from ..abstract_learner import AbstractLearner

# TODO check if there exists any ready function for this
def join_batch(b1, b2):
    b = [torch.cat([t1,t2]) for t1, t2 in zip(b1, b2)]
    return tuplify(b)

# TODO check if there exists any ready function for this
def divide_batch(b):
    b1 = [t[:t.size(0) // 2] for t in b]
    b2 = [t[t.size(0) // 2:] for t in b]
    return tuplify(b1), tuplify(b2)

class AbstractGANLearner(AbstractLearner):
    """
    This is a base class for all GANLearners,
    which may have sufficiently different different education algorithms.
    """
    def __init__(self, dls, gen, dis, join=False, cbs=None):
        AbstractLearner.__init__(self,dls,model=None,cbs=cbs)
        self.dls,self.generator,self.discriminator,self.join = dls,gen,dis,join
    
    def gen_batch(self, b):
        i = getattr(self.dls, 'n_gen', 1)
        j = getattr(self.dls, 'n_dis', 1)
        return (*b[:i], *b[i+j:])
    
    def dis_batch(self, b, gen_out=None):
        i = getattr(self.dls, 'n_gen', 1)
        j = getattr(self.dls, 'n_dis', 1)
        if gen_out is None:
            return b[i:]
        else:
            return (*tuplify(gen_out), *b[i+j:])
    
    def generate(self, gb):
        self.gb = gb # TODO
        self('before_gen')
        g = self.generator(*self.gb)
        self('after_gen')
        self.gb = None # TODO
        return tuplify(g)
    
    def discriminate(self, db):
        self.db = db # TODO
        self('before_dis')
        d = self.discriminator(*self.db)
        self('after_dis')
        self.db = None # TODO
        return tuplify(d)
    
    def discriminate_x2(self, real_db, fake_db):
        """
        With normalization, the fake and real should be
        in the same batch to avoid disparate statistics.
        """
        if self.join:
            db = join_batch(real_db, fake_db)
            pred = self.discriminate(db)
            r_pred, f_pred = divide_batch(pred)
        else:
            r_pred = self.discriminate(real_db)
            f_pred = self.discriminate(fake_db)
        return r_pred, f_pred
    
    # TODO check if this is correct, optimal and so on
    def _set_device(self, b):
        model = self.generator
        model_device = torch.device(torch.cuda.current_device()) if next(model.parameters()).is_cuda else torch.device('cpu')
        dls_device = getattr(self.dls, 'device', default_device())
        if model_device == dls_device: return to_device(b, dls_device)
        else: return to_device(b, model_device)
