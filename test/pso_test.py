import pyswarms as ps
import numpy as np
import torch
import jsonlines
import heapq # 导入堆模块
import time
import sys
import csv
from draw import drawSimpleImg,drawDoubledata

sys.path.append("./model")
from unixcoder import init_unixcoder
# to-do list
# 1. batch处理torch.einsum - ( nl:code -> 1:n/n:1 )
# 2. 用GPU多进程处理
# 3. 用CPU多线程/多进程处理
# 4. GPU & CPU 同时处理
# 5. 添加其他功能

# 普通搜索
def print_res(res):
    print('---------this is the top 10 result---------')
    for i in res:
        print(i['score'])

def get_search_file(language):
    return f'./src/search_source/{language}/search_source_{language}_score_first.jsonl'

def initNL(model,device,nl_str): 
    # Encode NL
    tokens_ids = model.tokenize([nl_str],max_length=512,mode="<encoder-only>")
    source_ids = torch.tensor(tokens_ids).to(device)
    tokens_embeddings,nl_embedding = model(source_ids)
    norm_nl_embedding = torch.nn.functional.normalize(nl_embedding, p=2, dim=1)
    return norm_nl_embedding

def myTorchEinsum(norm_code_embedding,norm_nl_embedding):
    code_nl_similarity = torch.einsum("ac,bc->ab",norm_code_embedding,norm_nl_embedding)
    return code_nl_similarity

# 数据读入缓存测试
def cache_get_source_data(language,searchNum=55000):
    sourceFile = get_search_file(language)
    sourceData = []
    try:
        with jsonlines.open(sourceFile, mode='r') as reader:
            for row,i in zip(reader,range(0,searchNum+1)):
                sourceData.append(row)
    except Exception as e:
        print(e)
        print(f"{language} cache all data")

    return sourceData

# 多重二分搜索+剪枝
def prune_search(arr, target,index):
    left, right = 0, len(arr)-1
    res = -1
    while left <= right:
        mid = (left + right) // 2
        if arr[mid]['norm_code_embedding'][0][index] >= target:
            res = mid
            right = mid - 1
        else:
            left = mid + 1
    if res == -1:
        res = len(arr) - 1
    return res

def get_index(sourceData,norm_nl_embedding,index,deviation,searchNum,useSearchNum=False):
    target=norm_nl_embedding[0][index].item()
    if(useSearchNum):
        mid=prune_search(sourceData,target,index)
        left=int(mid-searchNum/2) if int(mid-searchNum/2)>=0 else 0
        right = int(mid+searchNum/2) if int(mid+searchNum/2)<=len(sourceData)-1 else len(sourceData)-1
    else:
        left=prune_search(sourceData,target-deviation,index)
        right=prune_search(sourceData,target+deviation,index)
    return left,right

def prune_cache_getBestMatch(norm_nl_embedding,heap,device,sourceData,left,right,index,drawImg=False):
    if (drawImg):
        score_list=[]
        distant_list=[]

    try:
        for i in range(left,right+1):
            # print(f'i={i}')
            row=sourceData[i]
            norm_code_embedding = torch.Tensor(row["norm_code_embedding"]).to(device)
            # Normalize embedding
            code_nl_similarity = myTorchEinsum(norm_code_embedding,norm_nl_embedding)
            
            if (drawImg):
                score_list.append(code_nl_similarity.item())
                distant_list.append(abs(norm_code_embedding[0][index].item()-norm_nl_embedding[0][index].item())*10)

            if(row["func_name"]==""):
                row["func_name"]="None"
            resourcCode=[row['func_name'],row["norm_code_embedding"],row['original_string'],row['url']]
            # 压入堆中
            if len(heap) < 10: # 如果堆中元素个数小于10
                heapq.heappush(heap, (code_nl_similarity.item(), resourcCode)) # 直接把code_nl_similarity作为一个元组压入堆中
            elif code_nl_similarity > heap[0][0]: # 如果code_nl_similarity的第一个元素大于堆顶元素的第一个元素
                heapq.heappop(heap) # 弹出堆顶元素
                heapq.heappush(heap, (code_nl_similarity.item(), resourcCode)) # 把code_nl_similarity作为一个元组压入堆中
    except Exception as e:
        print(e)
        print("get_index error???")

    if (drawImg):
        drawDoubledata(score_list,distant_list)

def prune_cache_code_search(model,device,sourceData,searchNum,nl_str,deviation,binaryTime,useSearchNum=False,drawImg=False): 
    # start =time.perf_counter()

    norm_nl_embedding=initNL(model,device,nl_str)

    heap = [] # 创建一个空的堆
    heapq.heapify(heap) # 把列表转换成最小堆

    index=0
    left,right=get_index(sourceData,norm_nl_embedding,index,deviation,searchNum,useSearchNum)
    
    for index in range(1,binaryTime):
        print(f'id of sourceData:{id(sourceData)},len of sourceData:{len(sourceData)}')
        sourceData=sourceData[left:right]
        sourceData.sort(key=lambda x:x['norm_code_embedding'][0][index],reverse=False)
        left,right=get_index(sourceData,norm_nl_embedding,index,deviation,searchNum,useSearchNum)

    prune_cache_getBestMatch(norm_nl_embedding,heap,device,sourceData,left,right,index,drawImg)
    
    res = []
    for i in range(0, len(heap)):
        tuple = heapq.heappop(heap)
        res.insert(0,{"score":tuple[0],"func_name":tuple[1][0],"origin_code":tuple[1][1],"url":tuple[1][2]})
        # print(tuple[0],tuple[1][0]) 

    # end = time.perf_counter()
    # print('Running time: %s Seconds'%(end-start))   

    return res 








# 粒子群算法
def pso_fitness(x,norm_nl_embedding,sourceData,device):
    # print(f'\n-------x={x},type={type(x)}----\n')
    res=np.empty(len(x))

    for i in range(0,len(x)):
        norm_code_embedding = torch.Tensor(sourceData[int(np.round(x[i])[0])]["norm_code_embedding"]).to(device)
        code_nl_similarity = myTorchEinsum(norm_code_embedding,norm_nl_embedding)
        res[i]=1-code_nl_similarity.item()

    # print(f'\n------res={res},type={type(res)}----\n')
    return res

def pso_prune_cache_getBestMatch(norm_nl_embedding,heap,device,sourceData,pso_param):
    # 定义粒子群算法参数和粒子数量和迭代次数
    options = {'c1': pso_param['options']['c1'], 'c2': pso_param['options']['c2'], 'w':  pso_param['options']['w']}
    n_particles=pso_param['n_particles']
    iters=pso_param['iters']
    n_processes=pso_param['n_processes']

    # 定义维度和搜索空间
    n_dims = 1
    bounds = ([0],[len(sourceData)-1])

    # 初始化粒子群算法优化器
    optimizer = ps.single.GlobalBestPSO(n_particles=n_particles, dimensions=n_dims, bounds=bounds, options=options)
    # 运行粒子群算法优化器
    best_cost, best_pos = optimizer.optimize(pso_fitness, iters=iters, norm_nl_embedding=norm_nl_embedding,sourceData=sourceData,device=device)

    # 输出最优解和最优值
    print(f'Best position:{best_pos} , Best cost:{1-best_cost}')

    # 结果放入heap中
    res_index=int(np.round(best_pos)[0])
    resourcCode=[sourceData[res_index]["func_name"],sourceData[res_index]["norm_code_embedding"],sourceData[res_index]['original_string'],sourceData[res_index]['url']]
    for i in range(10):
        if len(heap) < 10: # 如果堆中元素个数小于10
            heapq.heappush(heap, (1-best_cost, resourcCode))
        elif 1-best_cost > heap[0][0]: # 如果code_nl_similarity的第一个元素5大于堆顶元素的第一个元素
            heapq.heappop(heap) # 弹出堆顶元素
            heapq.heappush(heap, (1-best_cost, resourcCode))

def pso_prune_cache_code_search(model,device,sourceData,searchNum,nl_str,deviation,binaryTime,pso_param,useSearchNum=False): 
    # start =time.perf_counter()

    norm_nl_embedding=initNL(model,device,nl_str)

    heap = [] # 创建一个空的堆
    heapq.heapify(heap) # 把列表转换成最小堆

    # 二分剪枝
    for index in range(0,binaryTime):
        sourceData.sort(key=lambda x:x['norm_code_embedding'][0][index],reverse=False)
        # prune_cache_getBestMatch(norm_nl_embedding,heap,device,sourceData,index,drawImg)
        left,right=get_index(sourceData,norm_nl_embedding,index,deviation,searchNum,useSearchNum)
        if((right-left)<10):
            break
        sourceData=sourceData[left:right]
        # print(f'id of sourceData:{id(sourceData)},len of sourceData:{len(sourceData)}')
        # prune_cache_getBestMatch(norm_nl_embedding,heap,device,sourceData,index,drawImg)
    # print(f'id of sourceData:{id(sourceData)},len of sourceData:{len(sourceData)}')
    # print(f'index {index}')
    pso_prune_cache_getBestMatch(norm_nl_embedding,heap,device,sourceData,pso_param)
    

    res = []
    for i in range(0, len(heap)):
        tuple = heapq.heappop(heap)
        res.insert(0,{"score":tuple[0],"func_name":tuple[1][0],"origin_code":tuple[1][1],"url":tuple[1][2]})
        # print(tuple[0],tuple[1][0]) 

    # end = time.perf_counter()
    # print('Running time: %s Seconds'%(end-start))   

    return res


if __name__ == '__main__':
    print("cuda is available : {}".format(torch.cuda.is_available()))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = torch.device("cpu")
    language = "python"
    model_unixcoder = init_unixcoder(device,"microsoft/unixcoder-base-nine")
    print("init finish")
    
    # nl_str_list = ["Returns an array of bounding boxes of human faces in a image","Adds properties for all fields in this protocol message type.","Make sure a DB specifier exists, creating it if necessary.","Return the datetime truncated to the precision of the provided unit.","Return True if the class is a date type."]
    nl_str_list = ["Make sure a DB specifier exists, creating it if necessary."]

    searchNum = 5000
    useSearchNum=False
    drawImg=False
    printRes=True
    deviation=0.1
    binaryTime=3

    pso_param = { "options":{'c1': 0.5, 'c2': 0.8, 'w': 1.5}  ,"n_particles":20 , "iters":50 ,'n_processes':1}

    # 缓存数据
    sourceData=cache_get_source_data(language,searchNum) # 注意，这里二分需要读取全部数据
    print("push to cache finish")

    # PSO
    start =time.perf_counter()
    for nl_str  in nl_str_list:
        res=pso_prune_cache_code_search(model_unixcoder,device,sourceData,searchNum,nl_str,deviation,binaryTime,pso_param,useSearchNum)
        if(printRes):
            print_res(res)
    end = time.perf_counter()
    print('pso_prune_cache_code_search Running time: %s Seconds'%(end-start))