"""Originally SecondByBorufsen in Axelrod library.

Portions of this code were taken from https://github.com/Axelrod-Python/Axelrod/blob/dev/axelrod/strategies/axelrod_second.py
Modified by T.J. Gaffney for use in openevolve.
"""

from axelrod.action import Action
from axelrod.player import Player

C, D = Action.C, Action.D


class Evolvo(Player):

    name = "Evolvo"

# EVOLVE-BLOCK-START
    """
    This player keeps track of the the opponent's responses to own behavior:

    - `cd_count` counts: Opponent cooperates as response to player defecting.
    - `cc_count` counts: Opponent cooperates as response to player cooperating.

    The player has a defect mode and a normal mode.  In defect mode, the
    player will always defect.  In normal mode, the player obeys the following
    ranked rules:

    1. If in the last three turns, both the player/opponent defected, then
       cooperate for a single turn.
    2. If in the last three turns, the player/opponent acted differently from
       each other and they're alternating, then change next defect to
       cooperate.  (Doesn't block third rule.)
    3. Otherwise, do tit-for-tat.

    Start in normal mode, but every 25 turns starting with the 27th turn,
    re-evaluate the mode.  Enter defect mode if any of the following
    conditions hold:

    - Detected random:  Opponent cooperated 7-18 times since last mode
      evaluation (or start) AND less than 70% of opponent cooperation was in
      response to player's cooperation, i.e.
      cc_count / (cc_count+cd_count) < 0.7
    - Detect defective:  Opponent cooperated fewer than 3 times since last mode
      evaluation.

    When switching to defect mode, defect immediately.  The first two rules for
    normal mode require that last three turns were in normal mode.  When starting
    normal mode from defect mode, defect on first move.
    """

    def __init__(self):
        super().__init__()
        self.cd_counts, self.cc_counts = 0, 0
        self.mutual_defect_streak = 0
        self.echo_streak = 0
        self.flip_next_defect = False
        self.mode = "Normal"

    def try_return(self, to_return):
        """
        We put the logic here to check for the `flip_next_defect` bit here,
        and proceed like normal otherwise.
        """

        if to_return == C:
            return C
        # Otherwise look for flip bit.
        if self.flip_next_defect:
            self.flip_next_defect = False
            return C
        return D

    def strategy(self, opponent: Player) -> Action:
        turn = len(self.history) + 1

        if turn == 1:
            return C

        # Update the response history.
        if turn >= 3:
            if opponent.history[-1] == C:
                if self.history[-2] == C:
                    self.cc_counts += 1
                else:
                    self.cd_counts += 1

        # Check if it's time for a mode change.
        if turn > 2 and turn % 25 == 2:
            coming_from_defect = False
            if self.mode == "Defect":
                coming_from_defect = True

            self.mode = "Normal"
            coops = self.cd_counts + self.cc_counts

            # Check for a defective strategy
            if coops < 3:
                self.mode = "Defect"

            # Check for a random strategy
            if (8 <= coops <= 17) and self.cc_counts / coops < 0.7:
                self.mode = "Defect"

            self.cd_counts, self.cc_counts = 0, 0

            # If defect mode, clear flags
            if self.mode == "Defect":
                self.mutual_defect_streak = 0
                self.echo_streak = 0
                self.flip_next_defect = False

            # Check this special case
            if self.mode == "Normal" and coming_from_defect:
                return D

        # Proceed
        if self.mode == "Defect":
            return D
        else:
            assert self.mode == "Normal"

            # Look for mutual defects
            if self.history[-1] == D and opponent.history[-1] == D:
                self.mutual_defect_streak += 1
            else:
                self.mutual_defect_streak = 0
            if self.mutual_defect_streak >= 3:
                self.mutual_defect_streak = 0
                self.echo_streak = 0  # Reset both streaks.
                return self.try_return(C)

            # Look for echoes
            # Fortran code defaults two turns back to C if only second turn
            my_two_back, opp_two_back = C, C
            if turn >= 3:
                my_two_back = self.history[-2]
                opp_two_back = opponent.history[-2]
            if (
                self.history[-1] != opponent.history[-1]
                and self.history[-1] == opp_two_back
                and opponent.history[-1] == my_two_back
            ):
                self.echo_streak += 1
            else:
                self.echo_streak = 0
            if self.echo_streak >= 3:
                self.mutual_defect_streak = 0  # Reset both streaks.
                self.echo_streak = 0
                self.flip_next_defect = True

            # Tit-for-tat
            return self.try_return(opponent.history[-1])
# EVOLVE-BLOCK-END
