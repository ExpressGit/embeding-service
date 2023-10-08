from typing import Optional
from fastapi import FastAPI
import logging
from fastapi import FastAPI, Request, Body
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from sentence_transformers import SentenceTransformer
from EmbeddingModelLoad import embeding_model_load
from LLMModelLoad import llm_model_load
import uvicorn
app = FastAPI() # 建立一個 Fast API application
# 默认初始化模型
embedding_object_model = embeding_model_load()
default_embdding_model = embedding_object_model.load_model('moka-ai/m3e-base')

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

# 参数格式
# {
#   "model": "m3e",
#   "input": ["laf是什么"]
# }
@app.post('/embeddings')
def embeddings():   
    model: str = Body('m3e', title='model_name', embed=True)
    input: Optional[str] = Body(..., title="texts", embed=True)
    print(model)
    embdding_model = default_embdding_model
    if model != 'm3e':
        if model in embeding_dict.keys():
            embdding_model = embedding_object_model.load_model(embeding_dict['model'])
    embeddings = embdding_model.encode(input)
    #print(embeddings)
    return embeddings.tolist()




if __name__ == '__main__':
    # uvicorn.run(app)
    uvicorn.run(app="main:app", host="0.0.0.0", port=6050, reload=False)