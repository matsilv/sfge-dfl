from .hamming_loss import HammingLoss
from .regret import Regret

LOSS_CLASSES = {'Hamming': HammingLoss, 'Regret': Regret}
