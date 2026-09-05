from os import environ
from argparse import ArgumentParser
from ConfigSpace import Configuration


# temporary patch for upstream bug in SMAC/CS
Configuration.__eq__ = lambda self, other: isinstance(other, Configuration) and dict(self) == dict(other)


if __name__ == '__main__':
    environ['TOKENIZERS_PARALLELISM'] = 'false'

    parser = ArgumentParser()
    parser.add_argument('-r', '--rounds', type=int, default=1)
    parser.add_argument('-w', '--workers', type=int, default=4)
    argv = parser.parse_args()

    from smac import MultiFidelityFacade, Scenario
    from smac.main.config_selector import ConfigSelector
    from smac.intensifier.hyperband_utils import determine_HB

    from spamscouter.trainer import Trainer, CS
    from spamscouter.settings import BaseSettings

    class ScouterSettings(BaseSettings):
        CONNECTOR = 'CACHE'
        cache_path = '.cache/'

    trainer = Trainer(ScouterSettings())
    trainer.initialize_hpo()

    trials_per_round = int(determine_HB(min_budget=trainer.min_budget, max_budget=trainer.max_budget)['trials_used'])
    trials = argv.rounds * trials_per_round

    scenario = Scenario(
        configspace=CS,
        min_budget=trainer.min_budget,
        max_budget=trainer.max_budget,
        deterministic=True,
        n_trials=trials,
        n_workers=argv.workers,
        output_directory='.smac/',
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
