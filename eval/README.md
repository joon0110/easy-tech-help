# Evaluation assets

`screenshots/` contains the 24 historical image fixtures from the previous screenshot direction. They are preserved for reference and are not loaded by text V1 training, inference or tests.

Active text seed splits live in `data/text/`. Use validation for development and reserve test for final evaluation after freezing settings. A full text benchmark runner and results are not implemented yet. The live tests under `tests/` are development smoke checks, not benchmark results.
