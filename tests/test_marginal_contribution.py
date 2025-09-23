import pandas as pd
from sklearn.ensemble import RandomForestRegressor

from explainable_ai.marginal_contribution import AleXai

def test_uai_feature_importance(create_diabetes):
    rf_reg = RandomForestRegressor()
    rf_reg.fit(create_diabetes.data.values, create_diabetes.target.values)
    data = pd.concat([create_diabetes.data, create_diabetes.target], axis=1)

    info_args = {'feature_importance': {"which_feature": data.columns.values[0]}}


    xai_ale = AleXai(info_args, rf_reg, data)
    xai_ale.generate_arguments()

