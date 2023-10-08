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
embeding_model_load('moka-ai/m3e-base')

@app.get("/users/{user_id}") # 指定 api 路徑 (get方法)
def read_user(user_id: int, q: Optional[str] = None):
    return {"user_id": user_id, "q": q}

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
    model_encode = SentenceTransformer('moka-ai/m3e-base')
    embeddings = model_encode.encode(input)
    #print(embeddings)
    return embeddings.tolist()




if __name__ == '__main__':
    # uvicorn.run(app)
    uvicorn.run(app="main:app", host="127.0.0.1", port=8080, reload=True)