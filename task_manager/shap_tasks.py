import pickle
from .celery_app_xai import app_celery
from celery.utils.log import get_task_logger
from explainable_ai.explainability import ShapValuesExplanation
from utils.load_resource import XaiLoadResource, _get_datasource, _get_model


logger = get_task_logger(__name__)

@app_celery.task(bind=True)
def shap_exec(self, explanation_id, feature_importance, task_type, uri_datasource, uri_model):

    logger.info(f"SHAP_EXEC: running explanation (id={explanation_id}) (task_id={self.request.id})")

    df = _get_datasource(uri_datasource)
    load_me = _get_model(uri_model)
    model = pickle.loads(load_me)

    xai_shap = ShapValuesExplanation({'feature_importance': feature_importance}, 
                                     model, df, task_id=self.request.id)

    xai_shap.generate_arguments()
    return xai_shap.generated_args_dict | {"status":"IMAGE,RAW"}