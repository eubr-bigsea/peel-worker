import base64
import io
import pickle
from .celery_app_xai import app_celery
from celery.utils.log import get_task_logger
from utils.load_resource import XaiLoadResource, _get_datasource, _get_model
import matplotlib.pyplot as plt
import numpy as np
import os

from explainable_ai.explainability import GeneticProgrammingExplainer
from explainable_ai.plot_generate_xai import PltGenerate

logger = get_task_logger(__name__)


@app_celery.task(bind=True)
def gp_xai_exec(self, explanation_id, feature_importance, task_type, uri_datasource, uri_model):

    logger.info(f"SHAP_EXEC: running explanation (id={explanation_id}) (task_id={self.request.id})")

    df = _get_datasource(uri_datasource)
    load_me = _get_model(uri_model)
    model = pickle.loads(load_me)

    gpx = GeneticProgrammingExplainer({'feature_importance': feature_importance},
                                      model, df, gp_solver='operon', task_id=self.request.id)
    gpx.generate_arguments()

    # type sympy.float numbers to np.float64
    gpx.generated_args_dict['feature_importance'] = ([np.float64(i) for i in gpx.generated_args_dict['feature_importance'][0]],
                                                    gpx.generated_args_dict['feature_importance'][1])
    
    return gpx.generated_args_dict | {"status":"RAW"}
    ''' HARDCODED PLOT OF TREE EXPLAINER 
    AND BAR GRAPH OF FEATURE IMPORTANCE

    t = gpx.explainer.show_tree(is_base64=True)

    decoded_data = base64.b64decode(t)
    image_stream = io.BytesIO(decoded_data)
    image = plt.imread(image_stream)
    plt.imshow(image)
    plt.axis('off')
    plt.savefig('storage/output/api_tree.png')
    plt.close()

    plt_xai = PltGenerate(gpx.generated_args_dict)
    p = plt_xai.create_plots()

    decoded_data = base64.b64decode(p)
    image_stream = io.BytesIO(decoded_data)
    image = plt.imread(image_stream)
    plt.imshow(image)
    plt.axis('off')
    plt.savefig('storage/output/api_partial_derivatives.png')
    plt.close()
    '''