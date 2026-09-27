"""Real-time Super Mario Bros. agent driven by the Jev System One decision model.

The package is organised as one module per phase of the decision pipeline:

``config``      typed settings, with the Jev credential kept out of every repr
``emulator``    the emulator boundary: frames, RAM, joypad actions, savestates, pacing
``perception``  the frame -> detections layer (deterministic, no network)
``state``       detections + RAM -> typed state, and its Jev JSON projection
``trajectory``  candidate actions rolled out through the real emulator
``jev``         the Jev HTTP client and the typed question contracts
``advisor``     the optional local PlayJev advisor (never on the critical path)
``policy``      decision cadence, confidence gating, safety fallbacks, execution loop
``journal``     per-tick evidence, JSONL trace, annotated frame artifacts
``dashboard``   the live view of all of the above
"""

__version__ = "0.1.0"
