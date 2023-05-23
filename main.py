import uvicorn as uvicorn
from fastapi import FastAPI
from typing import Union
from pydantic import BaseModel

from enums.code_language import all_language,search_language
from service.cache_data import Cache_data
from service.code_search import *
from model.unixcoder import init_unixcoder
from service.singleton_model import Singleton_model


class input_entity(BaseModel):
    language: all_language
    description: str
    '''
    其他参数
    '''
    deviation: float = 0.1
    binaryTime: int = 10

    searchNum: int = 5000
    useSearchNum: bool = False
    # useSearchNum: Union[bool, None] = False

 
app_api = FastAPI()  # 必须实例化该类，启动的时候调用
# https://cloud.tencent.com/developer/article/1878630
device = 'cpu'
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
singleton_model = Singleton_model(device)
model_unixcoder = singleton_model.get_model(model_name="unixcoder")
# model_unixcoder = init_unixcoder(device,"microsoft/unixcoder-base-nine")
print("model_unixcoder init finish")
cache_data = Cache_data(20000)
print("cache_data init finish")



 
# 请求根目录
@app_api.get('/')
async def index():
    return {'message': '欢迎来到FastApi 服务！查看接口说明请访问/docs'}

@app_api.get('/get/code-search')
async def index(searchNum: int = 1000):
    language='python'
    nl_str = "Returns an array of bounding boxes of human faces in a image"
    res =code_search(model_unixcoder,device,language,searchNum,nl_str)
    return res



@app_api.post('/post/code-search')
def code_search_function(input: input_entity):
    language=input.language.value
    searchNum = input.searchNum
    nl_str = input.description
    res = code_search(model_unixcoder,device,language,searchNum,nl_str)
    return res

@app_api.post('/post/async/code-search')
async def code_search_function(input: input_entity):
    language=input.language.value
    searchNum = input.searchNum
    nl_str = input.description
    res = await async_code_search(model_unixcoder,device,language,searchNum,nl_str)
    return res


@app_api.post('/post/code-search/cache/prune')
def code_search_cache_prune(input: input_entity):
    sourceData=cache_data.get_data(input.language.value)
    nl_str = input.description
    deviation = input.deviation
    binaryTime = input.binaryTime
    searchNum = input.searchNum
    useSearchNum = input.useSearchNum
    res =multi_binary_cache_code_search(model_unixcoder,device,sourceData,searchNum,nl_str,deviation,binaryTime,useSearchNum)
    return res

@app_api.post('/post/async/code-search/cache/prune')
async def async_code_search_cache_prune(input: input_entity):
    sourceData=cache_data.get_data(input.language.value)
    nl_str = input.description
    deviation = input.deviation
    binaryTime = input.binaryTime
    searchNum = input.searchNum
    useSearchNum = input.useSearchNum
    res = await async_multi_binary_cache_code_search(model_unixcoder,device,sourceData,searchNum,nl_str,deviation,binaryTime,useSearchNum)
    return res

@app_api.post('/post/unasync/code-search/cache/prune')
async def unasync_code_search_cache_prune(input: input_entity):
    sourceData=cache_data.get_data(input.language.value)
    nl_str = input.description
    deviation = input.deviation
    binaryTime = input.binaryTime
    searchNum = input.searchNum
    useSearchNum = input.useSearchNum
    res = res =multi_binary_cache_code_search(model_unixcoder,device,sourceData,searchNum,nl_str,deviation,binaryTime,useSearchNum)
    return res



if __name__ == '__main__':
    # uvicorn.run(app="main:app_api", host="192.168.192.33", port=11341, reload=True)
    # uvicorn.run(app="main:app_api", host="192.168.192.33", port=11341, workers=5)
    uvicorn.run(app_api, host="192.168.192.33", port=11341)