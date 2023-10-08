"""
Conversation prompt templates.

We kindly request that you import fastchat instead of copying this file if you want to use it.
You can contribute back the changes you want to make.
"""

import dataclasses
from sentence_transformers import SentenceTransformer
class embeding_model_load(object):
    
    def __init__(self) -> None:
        pass
    
    def load_model(self,model_name_or_path):
        model = SentenceTransformer(model_name_or_path)
        return model
    
if __name__ == '__main__':
    model_name = 'moka-ai/m3e-base'
    embedding_object_model = embeding_model_load()
    embedding_model = embedding_object_model.load_model(model_name)
    input = ['hello 你好']
    input_vetor = embedding_model.encode(input)
    print(input_vetor)