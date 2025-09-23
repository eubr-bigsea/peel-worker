import os
from .understand_ai import Understanding
from alibi.explainers import ALE, plot_ale
import matplotlib.pyplot as plt

class AleXai(Understanding):

    def __init__(self, arguments_used, model_to_understand, data_source, task_id=None):
        super().__init__(arguments_used, model_to_understand, data_source, task_id=task_id)

        self.ale = ALE(self.model_to_understand.predict,
                       feature_names=self.data_source.columns.values,
                       target_names=[self.data_source.columns.values[-1]])

        x_ale = self.data_source[self.feature_names[:-1]].values
        self.ale_explained = self.ale.explain(x_ale)


    def _uai_feature_importance(self, *args, **kwargs):
        which_feature = kwargs.get('which_feature')

        plt.rcParams['figure.constrained_layout.use'] = True
        if which_feature is not None:
            plot_ale(self.ale_explained, features=[which_feature])
            if not os.path.exists('storage/output'):
                os.mkdir('storage/output')
            plt.savefig(f'storage/output/{self.task_id}.png')
            plt.close()
            #plt.show()
