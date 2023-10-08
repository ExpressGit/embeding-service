"""
Conversation prompt templates.

We kindly request that you import fastchat instead of copying this file if you want to use it.
You can contribute back the changes you want to make.
"""

import dataclasses
import torch
from sentence_transformers import SentenceTransformer
class embeding_model(object):
    
    def __init__(self) -> None:
        pass
    
    def load_model(self,model_name_or_path):
        # 预加载模型
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') # 检测是否有GPU可用，如果有则使用cuda设备，否则使用cpu设备
        if torch.cuda.is_available():
            print('本次加载模型的设备为GPU: ', torch.cuda.get_device_name(0))
        else:
            print('本次加载模型的设备为CPU.')
        model = SentenceTransformer(model_name_or_path,device=device)
        # model = SentenceTransformer(model_name_or_path)
        return model
    
if __name__ == '__main__':
    model_name = 'moka-ai/m3e-base'
    embedding_object_model = embeding_model()
    embedding_model = embedding_object_model.load_model(model_name)
    input = ['hello 你好']
    input_vetor = embedding_model.encode(input)
    print(input_vetor)