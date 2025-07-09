import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import json
from pathlib import Path
from spamscouter.trainer import CS
from ConfigSpace import CategoricalHyperparameter


if __name__ == '__main__':
    df = pd.DataFrame(columns=['loss', 'budget', *CS.keys()])

    for path in Path('.smac/').glob('*/*/runhistory.json'):
        with open(path) as fp:
            history = json.load(fp)

        for run in history['data']:
            if run['status'] == 1:
                config = history['configs'][str(run['config_id'])]
                df = df._append({
                    'loss': run['cost'],
                    'budget': run['budget'],
                    **{key: config[key] for key in CS.keys()},
                }, ignore_index=True)

    df['budget'] = df['budget'].round().astype(int)
    df['message_processing_method'] = df['message_processing_method'].astype('category')

    for budget in df['budget'].unique():
        selected = df[df['budget'] == budget]

        for key, value in CS.items():
            if isinstance(value, CategoricalHyperparameter):
                plot_fn = sns.violinplot
            else:
                plot_fn = sns.scatterplot

            plt.title(f'Single parameter analysis at budget of {budget}:')
            plot_fn(data=selected, x=key, y='loss')
            plt.show()
