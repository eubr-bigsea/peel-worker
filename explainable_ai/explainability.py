import pandas as pd
import shap
import numpy as np
import matplotlib.pyplot as plt
from explainer.gpx import GPX
from lime import lime_tabular
from .understand_ai import Understanding
import time
import os

class Explanation(Understanding):

    def __init__(self,
                 arguments_used,
                 model_to_understand=None,
                 data_source=None,
                 mode=None,
                 task_id=None):

        """

        :param arguments_used:
        :param model_to_understand:
        :param data_source:
        :param feature_names:
        :param target_name:
        """

        super().__init__(arguments_used, model_to_understand, data_source, task_id=task_id)
        self.mode = mode


class GeneticProgrammingExplainer(Explanation):

    def __init__(self, arguments_used, model_to_understand, data_source,
                 mode='classification', gp_solver='operon',  num_samples=500, task_id=None):
        super().__init__(arguments_used, model_to_understand, data_source, mode, task_id=task_id)
        self.num_samples = num_samples
        self.explainer = None
        self.gp_solver = gp_solver
        self.explainer = GPX(x=self.data_source[self.feature_names[:-1]],
                             y=self.data_source[self.feature_names[-1]],
                             model_predict=self.model_to_understand.predict,
                             gp_model=self.gp_solver,
                             noise_set_num_samples=self.num_samples,
                             diff_as_numpy=False,
                             feature_names=self.feature_names)

    @property
    def gp_solver(self):
        return self._gp_solver

    @gp_solver.setter
    def gp_solver(self, gp_solver):
        if gp_solver == 'gplearn':
            from gplearn.genetic import SymbolicRegressor
            gp_hyper_parameters = {'population_size': 20,
                                   'generations': 50,
                                   'stopping_criteria': 0.00001,
                                   'p_crossover': 0.5,
                                   'p_subtree_mutation': 0.2,
                                   'p_hoist_mutation': 0.1,
                                   'p_point_mutation': 0.2,
                                   'const_range': (-5.0, 5.0),
                                   'parsimony_coefficient': 0.01,
                                   'init_depth': (2, 3),
                                   'n_jobs': -1,
                                   'low_memory': True,
                                   'function_set': ('add', 'sub', 'mul', 'div')}
            self._gp_solver = SymbolicRegressor(**gp_hyper_parameters)
        elif gp_solver == "operon":
            from pyoperon.sklearn import SymbolicRegressor
            gp_hyper_parameters = {
                'local_iterations': 100,
                'allowed_symbols': 'add,sub,mul,aq,constant,variable',
                'generations': 100,
                'mutation_probability': 0.2,
                'crossover_probability': 1.0,
                'crossover_internal_probability': 0.9,
                'population_size': 100,
                'max_length': 15,
                'objectives': ['mse'],
                'max_depth': 15,
                'tournament_size': 10,
                'epsilon': 1e-20,
                'reinserter': 'keep-best',
                'offspring_generator': 'basic'
            }
            self._gp_solver = SymbolicRegressor(**gp_hyper_parameters)
        else:
            raise ValueError('Genetic Programming solver does not exist')

    def _uai_feature_importance(self, *args, **kwargs):
        instance = kwargs.get('instance')
        if instance is not None:
            self.explainer.instance_understanding(instance)
            names = []
            values = []
            mapping_names = {'X' + str(i+1): name for i,name in enumerate(self._feature_names)}

            for k, v in self.explainer.derivatives_generate(instance, as_numpy=False).items():
                names.append(mapping_names[k])
                values.append(v)

            return values, names

        else:

            raise ValueError(f"{self.__class__.__name__} must of a instance")


class LocalExplanation(Explanation):
    def __init__(self, arguments_used, model_to_understand,
                 data_source, mode='classification', task_id=None):
        super().__init__(arguments_used, model_to_understand, data_source, mode, task_id=task_id)

        self.mode = mode
        x_lime = self.data_source[self.feature_names[:-1]].values
        self.explainer = lime_tabular.LimeTabularExplainer(training_data=x_lime,
                                                           mode=self.mode,
                                                           feature_names=self.feature_names)

   #def _uai_generate_table(self, *args, **kwargs):
    def _uai_feature_importance(self, *args, **kwargs):
        instance = kwargs.get('instance')
        instance = np.array(instance)
        n_feature = kwargs.get('n_feature')

        if self.mode == 'classification':
            predict = self.model_to_understand.predict_proba
        elif self.mode == 'regression':
            predict = self.model_to_understand.predict
        else:
            raise ValueError(f'{self.__class__.__name__} class doesnt handle with {self.mode} type')

        if instance is not None and n_feature:
            return self.explainer.explain_instance(instance,
                                                   predict,
                                                   num_features=n_feature)
        else:
            return self.explainer.explain_instance(instance,
                                                   predict)


class ShapValuesExplanation(Explanation):
    def __init__(self, arguments_used, model_to_understand, data_source, max_samples=5000, task_id=None):
        super().__init__(arguments_used, model_to_understand, data_source, task_id=task_id)

        self.data_source.set_axis(self.feature_names, axis=1)

        x_shap = self.data_source[self.feature_names[:-1]]
        self.max_samples = max_samples

        background = shap.maskers.Independent(x_shap, max_samples=self.max_samples)
        self.explainer = shap.explainers.Exact(self.model_to_understand.predict, background)

    def _uai_feature_importance(self, *args, **kwargs):
        instance = kwargs.get("instance")
        
        if instance is None:
            raise ValueError(f"{self.__class__.__name__} class can' t execut method "
                             f"feature_importance with instance None")
        else:
            instance = pd.DataFrame([instance], columns=self.feature_names[:-1])

        shap_type_xai = kwargs.get("shap_type_xai")

        plt.rcParams['figure.constrained_layout.use'] = True
        if shap_type_xai == "waterfall":
            shap_values = self.explainer(instance)
            shap.plots.waterfall(shap_values[0], show=False)

        elif shap_type_xai == "bar":
            shap_values = self.explainer(instance)
            shap.plots.bar(shap_values[0], show=False)

        else:
            raise ValueError(f"{self.__class__.__name__} class doesnt know "
                             f"how to handle with shap_type: {shap_type_xai}")

        output_name = f"{self.task_id}.png"

        if not os.path.exists('storage/output'):
            os.mkdir('storage/output')
        plt.savefig(f"storage/output/{output_name}")
        plt.close()

        return shap_values.values[0].tolist(), self.feature_names

