"""
    Annealing methods used for the standard deviation of a gaussian probabilistic model.
"""

########################################################################################################################


class BaseAnnealer:
    """
    Abstract class to wrap the annealers.
    """

    def __init__(self, tot_epochs: int, min_anneal_val: float):
        """
        :param tot_epochs: int; epochs horizon for the annealing.
        :param min_anneal_val: float; minimum annealing factor (e.g. 0.1 is the 10% of the initial value).
        """
        self._tot_epochs = tot_epochs

        # Sanity check the minimum annealing factor must be in ]0, 1[
        assert 0 < min_anneal_val < 1

        self._min_anneal_val = min_anneal_val

    @property
    def get_tot_epochs(self) -> int:
        """
        :return: int; epochs horizon for the annealing.
        """
        return self._tot_epochs

    def get_annealing_factor(self) -> float:
        """
        :return: float; current annealing factor.
        """
        raise NotImplementedError()

########################################################################################################################


class LinearAnnealer(BaseAnnealer):
    def __init__(self, tot_epochs: int, min_anneal_val: float):
        """
        :param tot_epochs: int; epochs horizon for the annealing.
        :param min_anneal_val: float; minimum annealing factor (e.g. 0.1 is the 10% of the initial value).
        """
        super().__init__(tot_epochs, min_anneal_val)

    def get_annealing_factor(self, epoch: int) -> float:
        """
        The new value is computed as annealed_val = annealing_factor * current_val.
        :param epoch: int; the epoch index.
        :return: float; the annealing factor.
        """

        return max(1.0 - epoch / self._tot_epochs, self._min_anneal_val)
