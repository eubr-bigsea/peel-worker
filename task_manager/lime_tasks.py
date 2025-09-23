import pickle

from explainable_ai.explainability import LocalExplanation
from .celery_app_xai import app_celery
from celery.utils.log import get_task_logger
from utils.load_resource import _get_datasource, _get_model


logger = get_task_logger(__name__)

@app_celery.task(bind=True)
def lime_exec(self, explanation_id, feature_importance, task_type, uri_datasource, uri_model):

    logger.info(f"LIME_EXEC: running explanation (id={explanation_id}) (task_id={self.request.id})")
    df = _get_datasource(uri_datasource)
    load_me = _get_model(uri_model)
    model = pickle.loads(load_me)

    lime_uai = LocalExplanation({'feature_importance': feature_importance}, 
                                     model, df, mode=task_type, task_id=self.request.id)
    lime_uai.generate_arguments()
    result = lime_uai.generated_args_dict['feature_importance']
    
    return {'result':result.as_list()} | {"status":"RAW"}