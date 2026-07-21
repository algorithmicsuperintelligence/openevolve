import importlib.util
import traceback

import axelrod as axl

from openevolve.evaluation_result import EvaluationResult


def evaluate(program_path):
    """
    Evaluate the program by running it multiple times and checking how close
    it gets to the known global minimum.

    Args:
        program_path: Path to the program file

    Returns:
        Dictionary of metrics
    """
    try:
        # Load the program
        spec = importlib.util.spec_from_file_location("program", program_path)
        program = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(program)

        existing_players = [s() for s in axl.short_run_time_strategies]
        new_player = program.Evolvo()
        players = [new_player] + existing_players[:]
        # edges should be new_player against everyone else
        edges = [(0, i) for i in range(1, len(players))]
        tournament = axl.Tournament(players, edges=edges, repetitions=1)
        results = tournament.play(progress_bar=False)
        
        evolvo_score = None
        evolvo_artifacts = dict()
        for player in results.summarise():
            if player.Name == program.Evolvo.name:
                evolvo_score = player.Median_score
                evolvo_artifacts["Cooperation_rating"] = player.Cooperation_rating
                evolvo_artifacts["Wins"] = player.Wins
                evolvo_artifacts["CC_rate"] = player.CC_rate
                evolvo_artifacts["CD_rate"] = player.CD_rate
                evolvo_artifacts["DC_rate"] = player.DC_rate
                evolvo_artifacts["DD_rate"] = player.DD_rate
                evolvo_artifacts["CC_to_C_rate"] = player.CC_to_C_rate
                evolvo_artifacts["CD_to_C_rate"] = player.CD_to_C_rate
                evolvo_artifacts["DC_to_C_rate"] = player.DC_to_C_rate
                evolvo_artifacts["DD_to_C_rate"] = player.DD_to_C_rate
        assert(evolvo_score is not None)

        return EvaluationResult(
            metrics={
                "combined_score": evolvo_score,
                "error": 0.0,
            },
            artifacts=evolvo_artifacts,
        )

    except Exception as e:
        print(f"Evaluation failed completely: {str(e)}")
        print(traceback.format_exc())
        
        # Create error artifacts
        error_artifacts = {
            "error_type": type(e).__name__,
            "error_message": str(e),
            "full_traceback": traceback.format_exc(),
            "suggestion": "Check for syntax errors or missing imports in the generated code"
        }
        
        return EvaluationResult(
            metrics={
                "combined_score": 0.0,
                "error": str(e),
            },
            artifacts=error_artifacts,
        )
