from smac import MultiFidelityFacade, Scenario
from spamscouter.trainer import Trainer, CS
from spamscouter.settings import BaseSettings


class ScouterSettings(BaseSettings):
    CONNECTOR = 'CACHE'
    cache_path = '.cache/'


if __name__ == '__main__':
    trainer = Trainer(ScouterSettings())
    trainer.initialize_hpo()

    smac = MultiFidelityFacade(Scenario(
        configspace=CS,
        min_budget=trainer.min_budget,
        max_budget=trainer.max_budget,
        deterministic=True,
        # n_workers=10,
        output_directory='.smac/',
    ), trainer.train_and_validate)
    incumbent = smac.optimize()

    print(f'Best HP Configuration: {incumbent}')
