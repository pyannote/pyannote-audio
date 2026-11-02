# Changelog

## Version 4.1.0 (2026-10-14)

- BREAKING: drop Python 3.10 support; require Python 3.11 or newer
- feat(optimize): add multi-objective optimization and Pareto front support
- fix(optimize): switch to Optuna's supported journal file backend
- setup: update minimum Optuna version to 5.0.0 and use its recommended default
  sampler for both single- and multi-objective optimization
- fix(setup): add missing `scipy` dependency

## Version 4.0.0 (2025-09-09)

- feat(optimize): add option to pass keyword arguments to pipeline during optimization

## Version 4.0.0rc1 (2025-02-11)

- feat(optimize): add option to pass keyword arguments to pipeline during optimization
- BREAKING: drop support for `Python` < 3.10
- BREAKING: switch to native namespace package
- BREAKING: remove `pyannote.pipeline.blocks` submodule
- setup: switch to `uv`

## Version 3.1.2 (2025-02-07)

- fix: should_prune() takes no argument
- fix: make `Optimizer.best_loss` return infinity with no trial
- fix: fix missing `scipy` import (#54, @arxaqapi)
- feat: add "--use-filter" option to filter training/validation files

## Version 3.0.1 (2023-09-22)

- BREAKING(cli): switch to latest pyannote.database API
- feat: add "seed" parameter for reproducible optimization
- feat(cli): add "device" section in configuration file
- feat(cli): add "--registry" option for custom database loading
- feat(cli): add "--average-case" option to optimize for average case
- setup: switch to optuna 3.1+
- feat: add support for optuna Journal storage

## Version 2.3 (2022-06-16)

- BREAKING: optimize loss estimate upper bound instead of average (#42)
- feat: add tests and typing (#41, @hadware)
- feat: add ParamDict structured hyper-parameter (#40, @hadware)
- feat: set sub-pipeline "training" attribute recursively (#39)
- doc: fix various typos

## Version 2.2 (2021-12-10)

- feat: add Pipeline.instantiated attribute

## Version 2.1 (2021-09-15)

- feat: add Pipeline.training attribute

## Version 2.0 (2020-11-25)

- BREAKING: remove "direction" argument from Optimizer
- feat: add Pipeline.get_direction method (defaults to "minimize")
- feat: add progress bar in "apply" mode

## Version 1.5.2 (2020-06-26)

- feat: add show_progress option to control second progress bar
- improve: catch optuna's ExperimentalWarning

## Version 1.5.1 (2020-06-18)

- feat: add second progress bar to display trial internal progress
- fix: skip Frozen parameters in pipeline.instantiate (@PaulLerner)
- fix: switch to pyannote.database 4.0+ (@PaulLerner)
- setup: switch to optuna 1.4+ and pyannote.core 4.0+

## Version 1.5 (2020-04-01)

- feat: add "direction" parameter to Optimizer
- fix: fix support for in-memory optimization (when db is None)
- setup: switch to pyannote.database 3.0

## Version 1.4 (2020-03-10)

- feat: add option to bootstrap optimization with pretrained pipeline

## Version 1.3 (2020-01-27)

- BREAKING: write "apply" mode output to "train" subdirectory
- feat: store best loss value in "params.yml"
- fix: handle corner case in pyannote.pipeline.blocks.clustering
- fix: use YAML safe loader

## Version 1.2 (2019-06-26)

- feat: add support for callable preprocessors
- setup: switch to pyannote.core 3.0
- setup: switch to pyannote.database 2.2

## Version 1.1.1 (2019-04-09)

- fix: do not raise FileExistsError when output directory exists in `pyannote-pipeline apply`
- fix: skip evaluation of protocols without groundtruth in `
pyannote-pipeline apply`
- setup: switch to pyannote.database 2.1

## Version 1.1 (2019-03-20)

- feat: add export to RTTM format
- setup: switch to pyannote.database 2.0
- fix: fix "use_threshold" parameter in HAC block

## Version 1.0 (2019-02-05)

- feat: add support for pyannote.metrics (through `Pipeline.get_metric`)
- feat: add support for optuna trial pruning
- feat: keep track of processing & evaluation time

## Version 0.3 (2019-01-17)

- feat: switch to optuna backend
- feat: add "use_threshold" option to HAC pipeline
- BREAKING: update Pipeline API
- BREAKING: update Optimizer API
- BREAKING: remove tensorboard support (for now)

## Version 0.2.1 (2018-12-04)

- first public release
