from smac import MultiFidelityFacade, Scenario
from smac.main.config_selector import ConfigSelector
from smac.intensifier.hyperband_utils import get_n_trials_for_hyperband_multifidelity
from spamscouter.trainer import Trainer, CS
from spamscouter.settings import BaseSettings
from os import cpu_count
from sys import argv


class ScouterSettings(BaseSettings):
    CONNECTOR = 'CACHE'
    cache_path = '.cache/'


if __name__ == '__main__':
    trainer = Trainer(ScouterSettings())
    trainer.initialize_hpo()

    multiplier = int(argv[1])
    trials = get_n_trials_for_hyperband_multifidelity(
        min_budget=trainer.min_budget,
        max_budget=trainer.max_budget,
        total_budget=trainer.max_budget * multiplier,
    )

    scenario = Scenario(
        configspace=CS,
        min_budget=trainer.min_budget,
        max_budget=trainer.max_budget,
        deterministic=True,
        n_trials=trials,
        n_workers=cpu_count() - 1,
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
