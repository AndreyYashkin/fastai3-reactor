from fastcore.basics import store_attr
from ..callback import Callback_m

class GANAugmentation(Callback_m):
    """
    Augmentation pipeline from
    https://arxiv.org/pdf/2006.06676.pdf
    """
    def __init__(self, diff_tfms):
        store_attr('diff_tfms')
    
    def before_dis(self):
        for tfm in self.diff_tfms:
            self.learn.db = tfm(self.db)
