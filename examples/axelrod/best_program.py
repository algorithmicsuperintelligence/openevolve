from axelrod.action import Action
from axelrod.player import Player

C, D = Action.C, Action.D


class AdaptiveCycleBreakerTFT(Player):
    """
    An adaptive strategy that switches between cooperative and defective modes
    based on opponent behavior analysis.

    This player tracks the opponent's cooperation patterns:

    - `opp_coops` counts total opponent cooperations since last evaluation
    - `cc_counts` counts opponent cooperations in response to player cooperating
    - `cd_counts` counts opponent cooperations in response to player defecting

    The player operates in two modes:

    **Normal Mode:**
    Uses conditional cooperation with the following ranked rules:

    1. If mutual defection occurred for 2+ consecutive turns, cooperate once
       to attempt breaking the cycle.
    2. If an alternating pattern is detected for 2+ turns (player and opponent
       taking turns defecting), flag the next defection to be converted to
       cooperation.
    3. Otherwise, play tit-for-tat.

    **Defect Mode:**
    Always defects.

    Starting in normal mode, the player re-evaluates its mode every 15 turns.
    It switches to defect mode if either condition holds:

    - Opponent cooperated fewer than 3 times in the evaluation period
    - Less than 83% of opponent's cooperations were in response to player's
      cooperation, i.e. cc_counts / opp_coops < 0.83

    When transitioning from defect mode back to normal mode, the player defects
    on the first turn of normal mode. The special rules for mutual defection
    and alternating patterns only apply during normal mode operation.
    """

    name = "AdaptiveCycleBreakerTFT"
    classifier = {
        "memory_depth": 15,
        "stochastic": False,
        "long_run_time": False,
        "inspects_source": False,
        "manipulates_source": False,
        "manipulates_state": False,
    }

    def __init__(self):
        super().__init__()
        # Counters used for deciding mode
        self.opp_coops = 0
        self.cd_counts, self.cc_counts = 0, 0

        # Streak counters
        self.mutual_defect_streak = 0
        self.echo_streak = 0

        self.flip_next_defect = False
        self.mode = "Normal"

    def strategy(self, opponent: Player) -> Action:
        turn = len(self.history) + 1
        if turn == 1:
            return C

        # Update counters.
        if turn >= 3:
            if opponent.history[-1] == C:
                self.opp_coops += 1
                if self.history[-2] == C:
                    self.cc_counts += 1
                else:
                    self.cd_counts += 1

        # Check if it's time for a mode change.
        if turn > 1 and turn % 15 == 0:
            coming_from_defect = (self.mode == "Defect")

            self.mode = "Normal"
            if self.opp_coops < 3 or self.cc_counts / self.opp_coops < 0.83:
                self.mode = "Defect"

            # Clear counters
            self.opp_coops = 0
            self.cd_counts, self.cc_counts = 0, 0
            if self.mode == "Defect":
                self.mutual_defect_streak = 0
                self.echo_streak = 0
                self.flip_next_defect = False

            # Check this special case: if coming from defect mode, defect on first move
            if self.mode == "Normal" and coming_from_defect:
                return D

        # In Defect mode, just defect
        if self.mode == "Defect":
            return D
        assert self.mode == "Normal"

        # Update streak counters
        if self.history[-1] == D and opponent.history[-1] == D:
            self.mutual_defect_streak += 1
        else:
            self.mutual_defect_streak = 0

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
        
        # Special behavior for streaks
        if self.mutual_defect_streak >= 2:
            self.mutual_defect_streak = 0
            return C

        if self.echo_streak >= 2:
            self.echo_streak = 0
            self.flip_next_defect = True

        # Just do tit-for-tat
        if opponent.history[-1] == C:
            return C
        
        if self.flip_next_defect:
            self.flip_next_defect = False
            return C
        
        return D
