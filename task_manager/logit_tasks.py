import pickle

from explainable_ai.interpretability import LogisticRegressionInterpretation
from .celery_app_xai import app_celery
from celery.utils.log import get_task_logger
from utils.load_resource import _get_datasource, _get_model


logger = get_task_logger(__name__)

@app_celery.task(bind=True)
def logit_exec(self, explanation_id, feature_importance, task_type, uri_datasource, uri_model):

    logger.info(f"LOGIT_EXEC: running explanation (id={explanation_id}) (task_id={self.request.id})")

    df = _get_datasource(uri_datasource)
    load_me = _get_model(uri_model)
    model = pickle.loads(load_me)

    logit_uai = LogisticRegressionInterpretation({'feature_importance': feature_importance}, 
                                     model, df, task_id=self.request.id)
    logit_uai.generate_arguments()
    
    return logit_uai.generated_args_dict | {"status":"RAW"}