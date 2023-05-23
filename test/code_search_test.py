import torch
import jsonlines
import heapq # 导入堆模块
import time
import sys
import bisect
import tqdm
from draw import drawSimpleImg,drawDoubledata

sys.path.append("./model")
from unixcoder import init_unixcoder
# to-do list
# 1. batch处理torch.einsum - ( nl:code -> 1:n/n:1 )
# 2. 用GPU多进程处理
# 3. 用CPU多线程/多进程处理
# 4. GPU & CPU 同时处理
# 5. 添加其他功能

# 原始搜索
def origin_get_search_file(language):
    return f'./src/search_source/{language}/search_source_{language}_score_first.jsonl'

def origin_initStr(model,device,code_nl_str): 
    # Encode NL
    tokens_ids = model.tokenize([code_nl_str],max_length=512,mode="<encoder-only>")
    source_ids = torch.tensor(tokens_ids).to(device)
    tokens_embeddings,str_embedding = model(source_ids)
    norm_str_embedding = torch.nn.functional.normalize(str_embedding, p=2, dim=1)
    return norm_str_embedding

def origin_getBestMatch(model,norm_nl_embedding,heap,device,sourceFile,searchNum):
    with jsonlines.open(sourceFile, mode='r') as reader:
        try:
            for row,i in zip(reader,range(0,searchNum+1)):
                if(i>=searchNum):
                    break
                # resourcCode=[row['func_name'],row["norm_code_embedding"],row['original_string'],row['url']]
                norm_code_embedding = origin_initStr(model,device,row['original_string'])
                # Normalize embedding
                code_nl_similarity = myTorchEinsum(norm_code_embedding,norm_nl_embedding)
                if(row["func_name"]==""):
                    row["func_name"]="None"
                resourcCode=[row['func_name'],row["original_string"],row['url']]

                # 压入堆中
                if len(heap) < 10: # 如果堆中元素个数小于10
                    heapq.heappush(heap, (code_nl_similarity.item(), resourcCode)) # 直接把code_nl_similarity作为一个元组压入堆中
                elif code_nl_similarity > heap[0][0]: # 如果code_nl_similarity的第一个元素大于堆顶元素的第一个元素
                    heapq.heappop(heap) # 弹出堆顶元素
                    heapq.heappush(heap, (code_nl_similarity.item(), resourcCode)) # 把code_nl_similarity作为一个元组压入堆中
        except Exception as e:
            print(e)
            print("search all data")

def origin_code_search(model,device,language,searchNum,nl_str): 
    # start =time.perf_counter()
    sourceFile=origin_get_search_file(language)
    norm_nl_embedding=origin_initStr(model,device,nl_str)

    heap = [] # 创建一个空的堆
    heapq.heapify(heap) # 把列表转换成最小堆

    origin_getBestMatch(model,norm_nl_embedding,heap,device,sourceFile,searchNum)
    
    res = []
    for i in range(0, len(heap)):
        tuple = heapq.heappop(heap)
        res.insert(0,{"score":tuple[0],"func_name":tuple[1][0],"origin_code":tuple[1][1],"url":tuple[1][2]})
        # print(tuple[0],tuple[1][0]) 

    # end = time.perf_counter()
    # print('Running time: %s Seconds'%(end-start))   

    return res 



# 普通搜索
def print_res(res):
    print('---------this is the top 10 result---------')
    for i in res:
        print(f"score:{i['score']}\tfunc_name:{i['func_name']}")
    print('---------           end           ---------')

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

def getBestMatch(norm_nl_embedding,heap,device,sourceFile,searchNum,drawImg=False):
    if (drawImg):
        score_list=[]
        distant_list=[]

    with jsonlines.open(sourceFile, mode='r') as reader:
        try:
            for row,i in zip(reader,range(0,searchNum+1)):
                if(i>=searchNum):
                    break
                # resourcCode=[row['func_name'],row["norm_code_embedding"],row['original_string'],row['url']]
                norm_code_embedding = torch.Tensor(row["norm_code_embedding"]).to(device)
                # Normalize embedding
                code_nl_similarity = myTorchEinsum(norm_code_embedding,norm_nl_embedding)
                if(row["func_name"]==""):
                    row["func_name"]="None"
                resourcCode=[row['func_name'],row["original_string"],row['url']]

                if (drawImg):
                    score_list.append(code_nl_similarity.item())
                    distant_list.append(abs(norm_code_embedding[0][0].item()-norm_nl_embedding[0][0].item())*10)

                # 压入堆中
                if len(heap) < 10: # 如果堆中元素个数小于10
                    heapq.heappush(heap, (code_nl_similarity.item(), resourcCode)) # 直接把code_nl_similarity作为一个元组压入堆中
                elif code_nl_similarity > heap[0][0]: # 如果code_nl_similarity的第一个元素大于堆顶元素的第一个元素
                    heapq.heappop(heap) # 弹出堆顶元素
                    heapq.heappush(heap, (code_nl_similarity.item(), resourcCode)) # 把code_nl_similarity作为一个元组压入堆中
        except Exception as e:
            print(e)
            print("search all data")

    if (drawImg):
        drawDoubledata(score_list,distant_list)

def code_search(model,device,language,searchNum,nl_str,drawImg=False): 
    # start =time.perf_counter()
    sourceFile=get_search_file(language)
    norm_nl_embedding=initNL(model,device,nl_str)

    heap = [] # 创建一个空的堆
    heapq.heapify(heap) # 把列表转换成最小堆

    getBestMatch(norm_nl_embedding,heap,device,sourceFile,searchNum,drawImg)
    
    res = []
    for i in range(0, len(heap)):
        tuple = heapq.heappop(heap)
        res.insert(0,{"score":tuple[0],"func_name":tuple[1][0],"origin_code":tuple[1][1],"url":tuple[1][2]})
        # print(tuple[0],tuple[1][0]) 

    # end = time.perf_counter()
    # print('Running time: %s Seconds'%(end-start))   

    return res 


# 异步操作
async def async_getBestMatch(norm_nl_embedding,heap,device,sourceFile,searchNum):
    with jsonlines.open(sourceFile, mode='r') as reader:
        for row,i in zip(reader,range(0,searchNum+1)):
            if(i>=searchNum):
                break
            resourcCode=[row['func_name'],row["norm_code_embedding"],row['url']]
            # resourcCode=[row['func_name'],row["norm_code_embedding"],row['original_string'],row['url']]
            norm_code_embedding = torch.Tensor(resourcCode[1]).to(device)
            # Normalize embedding
            code_nl_similarity = myTorchEinsum(norm_code_embedding,norm_nl_embedding)

            # 压入堆中
            if len(heap) < 10: # 如果堆中元素个数小于10
                heapq.heappush(heap, (code_nl_similarity.item(), resourcCode)) # 直接把code_nl_similarity作为一个元组压入堆中
            elif code_nl_similarity > heap[0][0]: # 如果code_nl_similarity的第一个元素大于堆顶元素的第一个元素
                heapq.heappop(heap) # 弹出堆顶元素
                heapq.heappush(heap, (code_nl_similarity.item(), resourcCode)) # 把code_nl_similarity作为一个元组压入堆中

async def async_code_search(model,device,sourceFile,language,nl_str): 
    start =time.perf_counter()

    norm_nl_embedding=initNL(model,device,nl_str)
    sourceFile=get_search_file(language)

    heap = [] # 创建一个空的堆
    heapq.heapify(heap) # 把列表转换成最小堆

    await getBestMatch(norm_nl_embedding,heap,device,sourceFile,searchNum)
    
    res = []
    for i in range(0, len(heap)):
        tuple = heapq.heappop(heap)
        res.insert(0,{"score":tuple[0],"func_name":tuple[1][0],"url":tuple[1][2]})
        # print(tuple[0],tuple[1][0]) 

    end = time.perf_counter()
    print('Running time: %s Seconds'%(end-start))   

    return res 


# 数据读入缓存测试
def cache_get_source_data(language,searchNum=55000):
    sourceFile = get_search_file(language)
    sourceData = []
    try:
        with jsonlines.open(sourceFile, mode='r') as reader:
            for row,i in zip(reader,range(0,searchNum)):
                sourceData.append(row)
    except Exception as e:
        print(e)
        print(f"{language} cache all data")

    return sourceData

def cache_getBestMatch(norm_nl_embedding,heap,device,sourceData,searchNum,drawImg=False):
    if (drawImg):
        score_list=[]
        distant_list=[]
    try:
        for row,i in zip(sourceData,range(0,searchNum+1)):
            if(i>=searchNum):
                break
            norm_code_embedding = torch.Tensor(row["norm_code_embedding"]).to(device)
            # Normalize embedding
            code_nl_similarity = myTorchEinsum(norm_code_embedding,norm_nl_embedding)

            if (drawImg):
                score_list.append(code_nl_similarity.item())
                distant_list.append(abs(norm_code_embedding[0][0].item()-norm_nl_embedding[0][0].item())*10)

            if(row["func_name"]==""):
                row["func_name"]="None"
            resourcCode=[row['func_name'],row['original_string'],row['url']]
            # 压入堆中
            if len(heap) < 10: # 如果堆中元素个数小于10
                heapq.heappush(heap, (code_nl_similarity.item(), resourcCode)) # 直接把code_nl_similarity作为一个元组压入堆中
            elif code_nl_similarity > heap[0][0]: # 如果code_nl_similarity的第一个元素大于堆顶元素的第一个元素
                heapq.heappop(heap) # 弹出堆顶元素
                heapq.heappush(heap, (code_nl_similarity.item(), resourcCode)) # 把code_nl_similarity作为一个元组压入堆中
    except Exception as e:
        print(e)
        print("search all data")
    
    if (drawImg):
        drawDoubledata(score_list,distant_list)
    
def cache_code_search(model,device,sourceData,searchNum,nl_str,drawImg=False): 
    # start =time.perf_counter()

    norm_nl_embedding=initNL(model,device,nl_str)

    heap = [] # 创建一个空的堆
    heapq.heapify(heap) # 把列表转换成最小堆

    cache_getBestMatch(norm_nl_embedding,heap,device,sourceData,searchNum,drawImg)
    
    res = []
    for i in range(0, len(heap)):
        tuple = heapq.heappop(heap)
        res.insert(0,{"score":tuple[0],"func_name":tuple[1][0],"origin_code":tuple[1][1],"url":tuple[1][2]})
        # print(tuple[0],tuple[1][0]) 

    # end = time.perf_counter()
    # print('Running time: %s Seconds'%(end-start))   

    return res 


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

def prune_cache_getBestMatch(norm_nl_embedding,heap,device,sourceData,index,drawImg=False):
    if (drawImg):
        score_list=[]
        distant_list=[]

    try:
        for row in sourceData:
            norm_code_embedding = torch.Tensor(row["norm_code_embedding"]).to(device)
            # Normalize embedding
            code_nl_similarity = myTorchEinsum(norm_code_embedding,norm_nl_embedding)
            
            if (drawImg):
                score_list.append(code_nl_similarity.item())
                distant_list.append(abs(norm_code_embedding[0][index].item()-norm_nl_embedding[0][index].item())*10)

            if(row["func_name"]==""):
                row["func_name"]="None"
            resourcCode=[row['func_name'],row['original_string'],row['url']]
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
    
    print(f'id of sourceData:{id(sourceData)},len of sourceData:{len(sourceData)}')
    for index in range(0,binaryTime):
        sourceData.sort(key=lambda x:x['norm_code_embedding'][0][index],reverse=False)
        prune_cache_getBestMatch(norm_nl_embedding,heap,device,sourceData,index,drawImg)
        left,right=get_index(sourceData,norm_nl_embedding,index,deviation,searchNum,useSearchNum)
        if((right-left)<10):
            break
        sourceData=sourceData[left:right]
        print(f'id of sourceData:{id(sourceData)},len of sourceData:{len(sourceData)}')
        # prune_cache_getBestMatch(norm_nl_embedding,heap,device,sourceData,index,drawImg)
    print(f'id of sourceData:{id(sourceData)},len of sourceData:{len(sourceData)}')
    # print(f'index {index}')
    prune_cache_getBestMatch(norm_nl_embedding,heap,device,sourceData,index,drawImg)
    
    res = []
    for i in range(0, len(heap)):
        tuple = heapq.heappop(heap)
        res.insert(0,{"score":tuple[0],"func_name":tuple[1][0],"origin_code":tuple[1][1],"url":tuple[1][2]})
        # print(tuple[0],tuple[1][0]) 

    # end = time.perf_counter()
    # print('Running time: %s Seconds'%(end-start))   

    return res 

# batch组装算法优化
def batch_myTorchEinsum(norm_code_embedding,norm_nl_embedding):
    # print(f"norm_code_embedding.shape:{norm_code_embedding.shape}")
    # print(f"norm_nl_embedding.shape:{norm_nl_embedding.shape}")
    code_nl_similarity = torch.einsum("ac,bc->ab",norm_code_embedding,norm_nl_embedding)
    # print(f"code_nl_similarity.shape:{code_nl_similarity.shape}")
    return code_nl_similarity

def batch_prune_cache_getBestMatch(norm_nl_embedding,heap,device,sourceData,left,right,batchSize):
    try:
        for i in range(left,right+1,batchSize):
            # print(f'i={i}')
            row=sourceData[i]
            norm_code_embeddings = torch.Tensor(row["norm_code_embedding"]).to(device)
            # Normalize embedding
            batchEnd=i+batchSize if i+batchSize<=right+1 else right+1
            for code_index in range(i+1,batchEnd):
                norm_code_embeddings=torch.cat((norm_code_embeddings,torch.Tensor(sourceData[code_index]["norm_code_embedding"]).to(device)),dim=0)

            code_nl_similarity = batch_myTorchEinsum(norm_code_embeddings,norm_nl_embedding)

            for code_index,score_index in zip(range(i,batchEnd),range(0,batchSize)):
                row=sourceData[code_index]
                if(row["func_name"]==""):
                    row["func_name"]="None"
                resourcCode=[row['func_name'],row['original_string'],row['url']]
                # 压入堆中
                if len(heap) < 10: # 如果堆中元素个数小于10
                    heapq.heappush(heap, (code_nl_similarity[score_index].item(), resourcCode)) # 直接把code_nl_similarity作为一个元组压入堆中
                elif code_nl_similarity[score_index].item() > heap[0][0]: # 如果code_nl_similarity的第一个元素大于堆顶元素的第一个元素
                    heapq.heappop(heap) # 弹出堆顶元素
                    heapq.heappush(heap, (code_nl_similarity[score_index].item(), resourcCode)) # 把code_nl_similarity作为一个元组压入堆中
    except Exception as e:
        print(e)
        print("batch_prune_cache_getBestMatch error???")

def batch_prune_cache_code_search(model,device,sourceData,searchNum,nl_str,deviation,binaryTime,batchSize,useSearchNum=False): 
    # start =time.perf_counter()

    norm_nl_embedding=initNL(model,device,nl_str)

    heap = [] # 创建一个空的堆
    heapq.heapify(heap) # 把列表转换成最小堆

    # print(f'id of sourceData:{id(sourceData)},len of sourceData:{len(sourceData)}')
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
    left,right =0,len(sourceData)-1
    batch_prune_cache_getBestMatch(norm_nl_embedding,heap,device,sourceData,left,right,batchSize)
    
    res = []
    for i in range(0, len(heap)):
        tuple = heapq.heappop(heap)
        res.insert(0,{"score":tuple[0],"func_name":tuple[1][0],"origin_code":tuple[1][1],"url":tuple[1][2]})
        # print(tuple[0],tuple[1][0]) 

    # end = time.perf_counter()
    # print('Running time: %s Seconds'%(end-start))   

    return res 



if __name__ == '__main__':
    print("run test.py")
    # print("cuda is available : {}".format(torch.cuda.is_available()))
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # language = "python"
    
    # searchNum = 40000
    # useSearchNum=False
    # drawImg=False
    # printRes=True
    # deviation=0.5
    # binaryTime=1
    # batchSize=1000
    
    # # nl_str_list = ["Returns an array of bounding boxes of human faces in a image","Adds properties for all fields in this protocol message type.","Make sure a DB specifier exists, creating it if necessary.","Return the datetime truncated to the precision of the provided unit.","Return True if the class is a date type."]
    # nl_str_list = ["Make sure a DB specifier exists, creating it if necessary."]
    # model_unixcoder = init_unixcoder(device,"microsoft/unixcoder-base-nine")
    # print("init finish")


    # # # 普通时间测试
    # # start =time.perf_counter()
    # # for nl_str  in nl_str_list:
    # #     res=code_search(model_unixcoder,device,language,searchNum,nl_str,drawImg)
    # #     if(printRes):
    # #         print_res(res)
    # # end = time.perf_counter()
    # # print('code_search Running time: %s Seconds'%(end-start)) 

    # # 缓存数据
    # sourceData=cache_get_source_data(language,searchNum) # 注意，这里二分需要读取全部数据
    # print("push to cache finish")

    # # # 缓存数据时间测试
    # # start =time.perf_counter()
    # # for nl_str  in nl_str_list:
    # #     res=cache_code_search(model_unixcoder,device,sourceData,searchNum,nl_str,drawImg)
    # #     if(printRes):
    # #         print_res(res)
    # # end = time.perf_counter()
    # # print('cache_code_search Running time: %s Seconds'%(end-start))

    # # # 多重二分搜索剪枝+存数据时间测试
    # # while True:
    # #     try:
    # #         print('-----------------please input:-----------------')
    # #         deviation=float(input("deviation:"))
    # #         binaryTime=int(input("binaryTime:"))
    # #         if(deviation==0 or binaryTime==0):
    # #             break
    # #     except Exception as e:
    # #         print(e)
    # #         print("input error???")
    # #         continue

    # #     start =time.perf_counter()
    # #     for nl_str  in nl_str_list:
    # #         res=prune_cache_code_search(model_unixcoder,device,sourceData,searchNum,nl_str,deviation,binaryTime,useSearchNum,drawImg)
    # #         if(printRes):
    # #             print_res(res)
    # #     end = time.perf_counter()
    # #     print('len(sourceData):',len(sourceData))
    # #     print('prune_cache_code_search Running time: %s Seconds'%(end-start))

    # # 多重二分搜索剪枝+存数据时间测试
    # start =time.perf_counter()
    # for nl_str  in nl_str_list:
    #     res=prune_cache_code_search(model_unixcoder,device,sourceData,searchNum,nl_str,deviation,binaryTime,useSearchNum,drawImg)
    #     if(printRes):
    #         print_res(res)
    # end = time.perf_counter()
    # print('prune_cache_code_search Running time: %s Seconds'%(end-start))

    # # batch时间测试
    # start =time.perf_counter()
    # for nl_str  in nl_str_list:
    #     res=batch_prune_cache_code_search(model_unixcoder,device,sourceData,searchNum,nl_str,deviation,binaryTime,batchSize,useSearchNum)
    #     if(printRes):
    #         print_res(res)
    # end = time.perf_counter()
    # print('batch_prune_cache_code_search Running time: %s Seconds'%(end-start))

