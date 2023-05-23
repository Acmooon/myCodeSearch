import torch
from unixcoder import UniXcoder
import jsonlines
import heapq # 导入堆模块
import time

# to-do list
# 1. batch处理torch.einsum - ( nl:code -> 1:n/n:1 )
# 2. 用GPU多进程处理
# 3. 用CPU多线程/多进程处理
# 4. GPU & CPU 同时处理
# 5. 添加其他功能

def myTorchEinsum(norm_code_embedding,norm_nl_embedding):
    code_nl_similarity = torch.einsum("ac,bc->ab",norm_code_embedding,norm_nl_embedding)
    return code_nl_similarity

def initNL(device):
    
    # device = torch.device("cpu")
    model = UniXcoder("microsoft/unixcoder-base-nine")
    model.to(device)

    # Encode NL
    print("cuda is available : {}".format(torch.cuda.is_available()))
    nl = "Returns an array of bounding boxes of human faces in a image"
    tokens_ids = model.tokenize([nl],max_length=512,mode="<encoder-only>")
    source_ids = torch.tensor(tokens_ids).to(device)
    tokens_embeddings,nl_embedding = model(source_ids)
    norm_nl_embedding = torch.nn.functional.normalize(nl_embedding, p=2, dim=1)
    return norm_nl_embedding

def getBestMatch(norm_nl_embedding,heap,device,sourceFile,searchNum):
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
                heapq.heappush(heap, (code_nl_similarity, resourcCode)) # 直接把code_nl_similarity作为一个元组压入堆中
            elif code_nl_similarity > heap[0][0]: # 如果code_nl_similarity的第一个元素大于堆顶元素的第一个元素
                heapq.heappop(heap) # 弹出堆顶元素
                heapq.heappush(heap, (code_nl_similarity, resourcCode)) # 把code_nl_similarity作为一个元组压入堆中




if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = 'cpu'
    sourceFile = './search_source/python/search_source_python.jsonl'
    searchNum = 1000
    norm_nl_embedding=initNL(device)

    heap = [] # 创建一个空的堆
    heapq.heapify(heap) # 把列表转换成最小堆

    start =time.perf_counter()
    getBestMatch(norm_nl_embedding,heap,device,sourceFile,searchNum)
    end = time.perf_counter()
    
    for i in range(0, len(heap)):
        tuple = heapq.heappop(heap)
        print(tuple[0],tuple[1][0]) 

    print('Running time: %s Seconds'%(end-start))     
 