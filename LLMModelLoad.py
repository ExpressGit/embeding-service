import dataclasses

from enum import Enum

class ModelName(Enum):
    pass

class llm_model_load(object):
    
    def __init__(self,model_name_or_path) -> None:
        self.model_name_or_path = model_name_or_path
        pass
    
    
    def load_model(model_name):
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer("moka-ai/m3e-base")