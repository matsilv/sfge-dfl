from enum import Enum


class DistributionTypes(str, Enum):
    gaussian = 'gaussian'
    poisson = 'poisson'

