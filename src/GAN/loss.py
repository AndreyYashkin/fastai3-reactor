import torch
import torch.nn.functional as F

class GANLoss:
    def loss_fn(self):
        raise NotImplementedError
    
    def gen_loss_fn(self):
        raise NotImplementedError

    def dis_loss_fn(self):
        raise NotImplementedError

# TODO make sigmoid optional
class  CrossEntropyLoss(GANLoss):
    def __init__(self, modified=True):
        self.modified = modified
    
    def loss_fn(self):
        return lambda r_pred, f_pred: -(F.binary_cross_entropy_with_logits(r_pred, torch.ones_like(r_pred)) + F.binary_cross_entropy_with_logits(f_pred, torch.zeros_like(f_pred)))
    
    def gen_loss_fn(self):
        if self.modified:
            return lambda f_pred: F.binary_cross_entropy_with_logits(f_pred, torch.ones_like(f_pred))
        else: # train failure is likely
            return lambda f_pred: -F.binary_cross_entropy_with_logits(f_pred, torch.zeros_like(f_pred))
    
    def dis_loss_fn(self):
        return lambda r_pred, f_pred: F.binary_cross_entropy_with_logits(r_pred, torch.ones_like(r_pred)) + F.binary_cross_entropy_with_logits(f_pred, torch.zeros_like(f_pred))
        
class HingeLoss(GANLoss):
    def gen_loss_fn(self):
        return lambda f_pred: -f_pred.mean()

    def dis_loss_fn(self):
        return lambda r_pred, f_pred: F.relu(1 - r_pred).mean() + F.relu(1 + f_pred).mean()
