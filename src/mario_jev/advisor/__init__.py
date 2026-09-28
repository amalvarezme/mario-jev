"""The optional local PlayJev-0.8B advisor: designed to be one more feature, never
the authority.

PlayJev-0.8B reads the frame directly and returns a probability per move. It is
published at 0.32 agreement with the teacher on Infinite Mario — the second worst
of its ten games — so that distribution is *designed to be* an input to the state
and never the decision; the policy that will consume it lands in WU4 slice C and
is not written yet. Its 2.2 GB of weights are deliberately not downloaded.

The advisor is absent by default (``MARIO_ADVISOR`` unset), and its absence is
always *typed* rather than implied: :attr:`PlayJevAdvisor.absence` carries a
``model_id`` and a reason from a closed set, so the journal that WU5 will write
can say why the second opinion was missing. :meth:`PlayJevAdvisor.advise` never
raises an :class:`Exception` — a :class:`BaseException` such as
``KeyboardInterrupt`` is deliberately not swallowed — and the optional
``torch``/``transformers`` stack is imported only inside the real loader, so
importing this package is free even where neither is installed.
"""

from .playjev import (
    ABSENCE_REASONS,
    BACKEND,
    FRAME_DTYPE,
    FRAME_SHAPE,
    REASON_DISABLED,
    REASON_LOAD_FAILED,
    REASON_NO_MATCHING_MOVES,
    REASON_TORCH_MISSING,
    REASON_TRANSFORMERS_MISSING,
    REASON_WEIGHTS_MISSING,
    AdvisorAbsence,
    AdvisorAdvice,
    AdvisorLoadError,
    AdvisorMove,
    LoadedAdvisor,
    PlayJevAdvisor,
    default_loader,
    matching_scores,
    normalise_move_name,
    rank_moves,
    renormalise,
    top_move,
    validate_frame,
)

__all__ = [
    "ABSENCE_REASONS",
    "BACKEND",
    "FRAME_DTYPE",
    "FRAME_SHAPE",
    "REASON_DISABLED",
    "REASON_LOAD_FAILED",
    "REASON_NO_MATCHING_MOVES",
    "REASON_TORCH_MISSING",
    "REASON_TRANSFORMERS_MISSING",
    "REASON_WEIGHTS_MISSING",
    "AdvisorAbsence",
    "AdvisorAdvice",
    "AdvisorLoadError",
    "AdvisorMove",
    "LoadedAdvisor",
    "PlayJevAdvisor",
    "default_loader",
    "matching_scores",
    "normalise_move_name",
    "rank_moves",
    "renormalise",
    "top_move",
    "validate_frame",
]
