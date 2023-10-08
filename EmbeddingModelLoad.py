"""
Conversation prompt templates.

We kindly request that you import fastchat instead of copying this file if you want to use it.
You can contribute back the changes you want to make.
"""

import dataclasses

class embeding_model_load(object):
    
    def __init__(self,model_name_or_path) -> None:
        self.model_name_or_path = model_name_or_path
        pass
    
    
    def load_model(self):
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer("moka-ai/m3e-base")
        
        return model
    
if __name__ == '__main__':
    model_name = 'moka-ai/m3e-base'
    embedding_object_model = embeding_model_load(model_name)
    embedding_model = embedding_object_model.load_model()
    input = ['hello 你好']
    input_vetor = embedding_model.encode(input)
    print(input_vetor)