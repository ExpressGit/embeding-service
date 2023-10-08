from typing import Optional
from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from sentence_transformers import SentenceTransformer
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import tiktoken
import numpy as np
from scipy.interpolate import interp1d
from typing import List
from sklearn.preprocessing import PolynomialFeatures
from EmbeddingModelLoad import embeding_model
import torch
import os
app = FastAPI() # 建立一個 Fast API application
# 默认初始化模型
embedding_object_model = embeding_model()
# default_embdding_model = embedding_object_model.load_model('moka-ai/m3e-base')

@app.get("/users/{user_id}") # 指定 api 路徑 (get方法)
def read_user(user_id: int, q: Optional[str] = None):
    return {"user_id": user_id, "q": q}

embeding_dict = {
    'm3e':'moka-ai/m3e-base',
    'm3e_base':'moka-ai/m3e-base',
    'm3e_small':'moka-ai/m3e-base',
    'm3e_large':'moka-ai/m3e-large'
}

model_dict = {
    'chatglm-6b':'THUDM/chatglm2-6b'
}

@app.on_event("startup")
async def startup_event():
    app.state.embedding_model = embedding_object_model.load_model('moka-ai/m3e-base')


# class ModelLoader:
#     _instance = None

#     def __new__(cls):
#         if cls._instance is None:
#             cls._instance = super().__new__(cls)
#             # 加载模型代码
#             cls._instance.model = keras.models.load_model("path/to/model")
#         return cls._instance

# model_loader = ModelLoader()


# @app.post('/v1/embeddings')
# def embeddings(item: Item):   
#     embdding_model = app.state.embedding_model
#     if item.model != 'm3e':
#         if item.model in embeding_dict.keys():
#             embdding_model = embedding_object_model.load_model(embeding_dict[item.model])
#     embeddings = embdding_model.encode(item.input)
#     #print(embeddings)
#     return embeddings.tolist()




#环境变量传入
# sk_key = os.environ.get('sk-key', 'sk-aaabbbcccdddeeefffggghhhiiijjjkkk')


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 创建一个HTTPBearer实例
security = HTTPBearer()



class EmbeddingRequest(BaseModel):
    input: List[str]
    model: str

class EmbeddingResponse(BaseModel):
    data: list
    model: str
    object: str
    usage: dict

def num_tokens_from_string(string: str) -> int:
    """Returns the number of tokens in a text string."""
    encoding = tiktoken.get_encoding('cl100k_base')
    num_tokens = len(encoding.encode(string))
    return num_tokens

# 插值法
def interpolate_vector(vector, target_length):
    original_indices = np.arange(len(vector))
    target_indices = np.linspace(0, len(vector)-1, target_length)
    f = interp1d(original_indices, vector, kind='linear')
    return f(target_indices)

def expand_features(embedding, target_length):
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

@app.post("/v1/embeddings", response_model=EmbeddingResponse)
async def get_embeddings(request: EmbeddingRequest):

    # if credentials.credentials != sk_key:
    #     raise HTTPException(
    #         status_code=status.HTTP_401_UNAUTHORIZED,
    #         detail="Invalid authorization code",
    #     )
    embdding_model = app.state.embedding_model
    # 计算嵌入向量和tokens数量
    embeddings = [embdding_model.encode(text) for text in request.input]

    # 如果嵌入向量的维度不为1536，则使用插值法扩展至1536维度
    # embeddings = [interpolate_vector(embedding, 1536) if len(embedding) < 1536 else embedding for embedding in embeddings]
    # 如果嵌入向量的维度不为1536，则使用特征扩展法扩展至1536维度
    embeddings = [expand_features(embedding, 1536) if len(embedding) < 1536 else embedding for embedding in embeddings]

    # Min-Max normalization
    # embeddings = [(embedding - np.min(embedding)) / (np.max(embedding) - np.min(embedding)) if np.max(embedding) != np.min(embedding) else embedding for embedding in embeddings]
    embeddings = [embedding / np.linalg.norm(embedding) for embedding in embeddings]
    # 将numpy数组转换为列表
    embeddings = [embedding.tolist() for embedding in embeddings]
    prompt_tokens = sum(len(text.split()) for text in request.input)
    total_tokens = sum(num_tokens_from_string(text) for text in request.input)


    response = {
        "data": [
            {
                "embedding": embedding,
                "index": index,
                "object": "embedding"
            } for index, embedding in enumerate(embeddings)
        ],
        "model": request.model,
        "object": "list",
        "usage": {
            "prompt_tokens": prompt_tokens,
            "total_tokens": total_tokens,
        }
    }


    return response

if __name__ == '__main__':
    # uvicorn.run(app)
    uvicorn.run(app="main:app", host="0.0.0.0", port=6050, reload=True,workers=2)