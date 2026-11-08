"""Originally Tit-for-Tat in Axelrod library.

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
    Cooperates on first move, and defects if the opponent defects.
    """

    def __init__(self):
        super().__init__()

    def strategy(self, opponent: Player) -> Action:
        # First move
        if not self.history:
            return C
        # React to the opponent's last move
        if opponent.history[-1] == D:
            return D
        return C
# EVOLVE-BLOCK-END
