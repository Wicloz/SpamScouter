from os import environ


for _threads in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    environ.setdefault(_threads, '1')
environ.setdefault('TOKENIZERS_PARALLELISM', 'false')


from argparse import ArgumentParser
from math import floor, ceil, log
from ConfigSpace import Configuration
import numpy as np


# temporary patch for upstream bug in SMAC/CS
Configuration.__eq__ = lambda self, other: isinstance(other, Configuration) and dict(self) == dict(other)


def trials_per_hyperband_round(min_budget, max_budget, eta=3):
    s_max = floor(log(max_budget / min_budget) / log(eta))

    return sum(
        floor(ceil(eta ** m * (s_max + 1) / (m + 1)) * eta ** -j)
        for m in range(s_max + 1)
        for j in range(m + 1)
    )


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('-r', '--rounds', type=int, default=None)
    parser.add_argument('-m', '--max-seconds', type=float, default=None)
    parser.add_argument('-w', '--workers', type=int, default=1)
    parser.add_argument('--cache', type=str, default='.cache/')
    parser.add_argument('--output', type=str, default='.smac/')
    parser.add_argument('-n', '--name', type=str, default=None)
    argv = parser.parse_args()

    if argv.rounds is None and argv.max_seconds is None:
        raise ValueError('Either --rounds or --max-seconds must be specified.')
    conditional_scenario_args = {}

    from smac import MultiFidelityFacade, Scenario
    from smac.main.config_selector import ConfigSelector
    from spamscouter.trainer import Trainer, CS
    from spamscouter.settings import BaseSettings

    class ScouterSettings(BaseSettings):
        CONNECTOR = 'CACHE'
        cache_path = argv.cache

    trainer = Trainer(ScouterSettings())
    trainer.initialize_hpo()

    if argv.max_seconds is not None:
        conditional_scenario_args['walltime_limit'] = argv.max_seconds
    else:
        conditional_scenario_args['walltime_limit'] = np.inf

    if argv.rounds is not None:
        conditional_scenario_args['n_trials'] = argv.rounds * trials_per_hyperband_round(trainer.min_budget, trainer.max_budget)
    else:
        conditional_scenario_args['n_trials'] = np.inf

    scenario = Scenario(
        configspace=CS,
        use_default_config=True,
        min_budget=trainer.min_budget,
        max_budget=trainer.max_budget,
        deterministic=True,
        n_workers=argv.workers,
        output_directory=argv.output,
        name=argv.name,
        **conditional_scenario_args,
    )

    config_selector = ConfigSelector(
        scenario=scenario,
        retrain_after=1,
        min_trials=len(CS) + 1,
    )

    smac = MultiFidelityFacade(
        scenario=scenario,
        target_function=trainer.train_and_validate,
        config_selector=config_selector,
    )

    incumbent = smac.optimize()
    print(f'Best HP Configuration: {incumbent}')
