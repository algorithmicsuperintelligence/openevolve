"""Originally OriginalGradual in Axelrod library.

Portions of this code were taken from https://github.com/Axelrod-Python/Axelrod/blob/dev/axelrod/strategies/titfortat.py
Modified by T.J. Gaffney for use in openevolve.
"""

from axelrod.action import Action
from axelrod.player import Player

C, D = Action.C, Action.D


class Evolvo(Player):

    name = "Evolvo"

# EVOLVE-BLOCK-START
    """
    A player that punishes defections with a growing number of defections
    but after punishing for `punishment_limit` number of times enters a calming
    state and cooperates no matter what the opponent does for two rounds.

    The `punishment_limit` is incremented whenever the opponent defects and the
    strategy is not in either calming or punishing state.
    """

    def __init__(self) -> None:

        super().__init__()
        self.calming = False
        self.punishing = False
        self.punishment_count = 0
        self.punishment_limit = 0

    def strategy(self, opponent: Player) -> Action:
        if self.calming:
            self.calming = False
            return C

        if self.punishing:
            if self.punishment_count < self.punishment_limit:
                self.punishment_count += 1
                return D
            else:
                self.calming = True
                self.punishing = False
                self.punishment_count = 0
                return C

        if D in opponent.history[-1:]:
            self.punishing = True
            self.punishment_count += 1
            self.punishment_limit += 1
            return D

        return C
# EVOLVE-BLOCK-END
