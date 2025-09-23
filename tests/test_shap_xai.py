from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingRegressor
# from ..explainable_ai.explainability import ShapValuesExplanation
import pandas as pd
from explainable_ai.explainability import ShapValuesExplanation

def test_shap_xai_waterfall(create_iris):
    dt_cls = DecisionTreeClassifier()
    dt_cls.fit(create_iris.data, create_iris.target)
    data = pd.concat([create_iris.data, create_iris.target], axis=1)

    a = data.iloc[[4]].values[0][:-1]

    arg_info = {"feature_importance": {"shap_type_xai": "waterfall", "instance": a}}

    xai_shap = ShapValuesExplanation(arg_info, dt_cls, data)
    xai_shap.generate_arguments()


def test_shap_xai_bar(create_iris):
    dt_cls = DecisionTreeClassifier()
    dt_cls.fit(create_iris.data, create_iris.target)
    data = pd.concat([create_iris.data, create_iris.target], axis=1)

    a = data.iloc[[4]].values[0][:-1]

    arg_info = {"feature_importance": {"shap_type_xai": "bar", "instance": a}}

    xai_shap = ShapValuesExplanation(arg_info, dt_cls, data)
    xai_shap.generate_arguments()




