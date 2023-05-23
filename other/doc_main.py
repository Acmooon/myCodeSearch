
res=0

# mian.py文件
import uvicorn as uvicorn
from fastapi import FastAPI

 
app_api = FastAPI()  # 必须实例化该类，启动的时候调用

 
# 请求路径
@app_api.get('/')
def index():
    """
    实现接口相应功能
    """

    # 返回数据
    return res


if __name__ == '__main__':
    #设置服务地址和端口，运行服务
    uvicorn.run(app="main:app_api", host="127.0.0.1", port=11341)
    