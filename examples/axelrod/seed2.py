"""Originally ForgetfulFoolMeOnce(0.05) in Axelrod library.

Portions of this code were taken from https://github.com/Axelrod-Python/Axelrod/blob/dev/axelrod/strategies/oncebitten.py
Modified by T.J. Gaffney for use in openevolve.
"""

import random

from axelrod.action import Action
from axelrod.player import Player

C, D = Action.C, Action.D


class Evolvo(Player):
    name = "Evolvo"

# EVOLVE-BLOCK-BEGIN
    """
    Forgives one D then retaliates forever on a second D. Sometimes randomly
    forgets the defection count, and so keeps a secondary count separate from
    the standard count in Player.
    """

    def __init__(self) -> None:
        """
        Parameters
        ----------
        forget_probability, float
            The probability of forgetting the count of opponent defections.
        """
        super().__init__()
        self.D_count = 0
        self._initial = C
        self.forget_probability = 0.05

    def strategy(self, opponent: Player) -> Action:
        r = random.random()
        if not opponent.history:
            return self._initial
        if opponent.history[-1] == D:
            self.D_count += 1
        if r < self.forget_probability:
            self.D_count = 0
        if self.D_count > 1:
            return D
        return C
# EVOLVE-BLOCK-END
