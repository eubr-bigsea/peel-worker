
from task_manager.celery_app_xai import app_celery

from task_manager.shap_tasks import shap_exec
from task_manager.tree_tasks import tree_exec
from task_manager.gpx_tasks import gp_xai_exec
from task_manager.ale_tasks import ale_exec
from task_manager.ensemble_tasks import ensemble_exec
from task_manager.logit_tasks import logit_exec
from task_manager.linear_tasks import linear_exec
from task_manager.lime_tasks import lime_exec

app = app_celery