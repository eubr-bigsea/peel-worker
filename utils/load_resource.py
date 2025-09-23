
import pandas as pd
import pickle
from pyarrow import fs

from celery.utils.log import get_task_logger


logger = get_task_logger(__name__)

def _get_model(uri_model):
        logger.info(uri_model)
        arrow = fs.LocalFileSystem()
        exists = arrow.get_file_info(uri_model).is_file

        if not exists:
            logger.error(f'Tried load model {uri_model}, but this model doesnt exist!')
            raise ValueError(f'Tried load model {uri_model}, but this model doesnt exist!!')
        
        with arrow.open_input_stream(uri_model) as stream:
            rd = stream.readall()
        return rd

def _get_datasource(uri_datasource):
    if uri_datasource.endswith(".csv"):
        df = pd.read_csv(uri_datasource, index_col=False)
        return df

class XaiLoadResource:

    def __init__(self, data_is_local=False, model_is_local=False):
        self.data_is_local = data_is_local
        self.model_is_local = model_is_local

    def get_model(self, model_name):
        if self.model_is_local:
            model_path = "/storage/models/" + model_name
            logger.info(model_path)
            arrow = fs.LocalFileSystem()
            exists = arrow.get_file_info(model_path).is_file
            if not exists:
                logger.error(f'class {self.__class__.__name__} tried load model {model_name}, '
                             f'but this model doesnt exist!!')
                raise ValueError(f'class {self.__class__.__name__} tried load model {model_name}, '
                                 f'but this model doesnt exist!!')
            with arrow.open_input_stream(model_path) as stream:
                rd = stream.readall()
            return rd

    def get_data(self, data_name):
        if self.data_is_local:
            if data_name.endswith(".csv"):
                data_path = '/storage/data/' + data_name
                df = pd.read_csv(data_path, index_col=False)
                return df
        else:
            logger.error(f"class {self.__class__.__name__} doesnt know how to load {data_name}")
            raise NotImplementedError(f"class {self.__class__.__name__} doesnt know how to load {data_name}")

