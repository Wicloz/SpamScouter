from os import environ
from argparse import ArgumentParser
from math import floor, ceil, log, sqrt
from ConfigSpace import Configuration


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
    environ['TOKENIZERS_PARALLELISM'] = 'false'

    parser = ArgumentParser()
    parser.add_argument('-r', '--rounds', type=int, default=1)
    parser.add_argument('-w', '--workers', type=int, default=4)
    parser.add_argument('--cache', type=str, default='.cache/')
    parser.add_argument('--output', type=str, default='.smac/')
    argv = parser.parse_args()

    from smac import MultiFidelityFacade, Scenario
    from smac.main.config_selector import ConfigSelector

    from spamscouter.trainer import Trainer, CS
    from spamscouter.settings import BaseSettings

    class ScouterSettings(BaseSettings):
        CONNECTOR = 'CACHE'
        cache_path = argv.cache

    trainer = Trainer(ScouterSettings())
    trainer.initialize_hpo()

    trials = argv.rounds * trials_per_hyperband_round(trainer.min_budget, trainer.max_budget)
    scenario = Scenario(
        configspace=CS,
        min_budget=trainer.min_budget,
        max_budget=trainer.max_budget,
        deterministic=True,
        n_trials=trials,
        n_workers=argv.workers,
        output_directory=argv.output,
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
