"""
Conversation prompt templates.

We kindly request that you import fastchat instead of copying this file if you want to use it.
You can contribute back the changes you want to make.
"""

import dataclasses
import torch
from sentence_transformers import SentenceTransformer
import tiktoken
from sklearn.preprocessing import PolynomialFeatures
from scipy.interpolate import interp1d
import numpy as np
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
    
    
    def num_tokens_from_string(self,string: str) -> int:
        """Returns the number of tokens in a text string."""
        encoding = tiktoken.get_encoding('cl100k_base')
        num_tokens = len(encoding.encode(string))
        return num_tokens

    # 插值法
    def interpolate_vector(self,vector, target_length):
        original_indices = np.arange(len(vector))
        target_indices = np.linspace(0, len(vector)-1, target_length)
        f = interp1d(original_indices, vector, kind='linear')
        return f(target_indices)

    def expand_features(self,embedding, target_length):
        poly = PolynomialFeatures(degree=2)
        expanded_embedding = poly.fit_transform(embedding.reshape(1, -1))
        expanded_embedding = expanded_embedding.flatten()
        if len(expanded_embedding) > target_length:
            # 如果扩展后的特征超过目标长度，可以通过截断或其他方法来减少维度
            expanded_embedding = expanded_embedding[:target_length]
        elif len(expanded_embedding) < target_length:
            # 如果扩展后的特征少于目标长度，可以通过填充或其他方法来增加维度
            expanded_embedding = np.pad(expanded_embedding, (0, target_length - len(expanded_embedding)))
        return expanded_embedding
    
if __name__ == '__main__':
    model_name = 'moka-ai/m3e-base'
    embedding_object_model = embeding_model()
    embedding_model = embedding_object_model.load_model(model_name)
    input = ['hello 你好']
    input_vetor = embedding_model.encode(input)
    print(input_vetor)