from code_search_test import *
from pso_test import pso_prune_cache_code_search
from tqdm import tqdm
import numpy as np
import csv
 
def print_time(start,end,name,len):
    if not isAvg:
        len=1
    runTime=(end-start)/len
    print(f'{name} Running time: {runTime} Seconds') 
    return runTime

def print_res(printRes,res,url):
    count_similarity=0
    accuracy=False
    score_threshold=0.3

    if printRes:
        print('--------- this is the top 10 result ---------')
        if(res[0]['url']==url or res[1]['url']==url or res[2]['url']==url):
            print('---------  o o o o o o o o o o o  ---------')
            accuracy=True
        else:
            print('---------  x x x x x x x x x x x  ---------')           
        for i in res:
            print(f"score:{i['score']}\tfunc_name:{i['func_name']}")
            if i['score']>score_threshold:
                count_similarity+=1
        print(f'score_threshold:{score_threshold}, count:{count_similarity/len(res)}')
        print('---------           end           ---------')
    
    else:
        if(res[0]['url']==url or res[1]['url']==url or res[2]['url']==url):
            accuracy=True
        for i in res:
            if i['score']>score_threshold:
                count_similarity+=1
    similarity_ratio=count_similarity/len(res)
    return [accuracy,similarity_ratio,res[0]['score']]

def print_count(statictic):
    len_test=len(statictic)
    count_accuracy=0
    count_similarity=0
    count_score=0
    for i in statictic:
        count_accuracy+=i[0]
        count_similarity+=i[1]
        count_score+=i[2]
    res_count={'test_num':len_test,'avg_accuracy':count_accuracy/len_test,'avg_similarity_ratio':count_similarity/len_test,'avg_best_score':count_score/len_test}
    i=res_count['test_num']
    print(f'test_num:{res_count["test_num"]}')
    print(f'avg_accuracy:{res_count["avg_accuracy"]}')
    print(f'avg_similarity_ratio:{res_count["avg_similarity_ratio"]}')
    print(f'avg_best_score:{res_count["avg_best_score"]}')
    return res_count
    
def get_nl_list(language,testNum,testStart=0):
    test_nl_list=[]
    testFile= f'./src/search_source/{language}/search_source_{language}_test.jsonl'
    with jsonlines.open(testFile, mode='r') as reader:
        for i in range(testStart):
            reader.read()
        for row,i in zip(reader,range(0,testNum)):
            test_nl_list.append([row['docstring'],row['url']])
            # print(f'func_name: {row["func_name"]}')
    return test_nl_list
            

def origin_experiment(model,device,language,searchNum,nl_str_list):
    statictic=[]
    start =time.perf_counter()
    for [nl_str,url] in tqdm(nl_str_list):
        res=origin_code_search(model,device,language,searchNum,nl_str)
        statictic.append(print_res(printRes,res,url))
    end = time.perf_counter()
    print_count(statictic)
    print_time(start,end,'origin_experiment',len(nl_str_list))

def normal_experiment(model,device,language,searchNum,nl_str_list):
    statictic=[]
    start =time.perf_counter()
    for [nl_str,url] in tqdm(nl_str_list):
        res=code_search(model,device,language,searchNum,nl_str,drawImg)
        statictic.append(print_res(printRes,res,url))
    end = time.perf_counter()
    print_count(statictic)
    print_time(start,end,'normal_experiment',len(nl_str_list))

def cache_experiment(model,device,sourceData,searchNum,nl_str_list):
    statictic=[]
    start =time.perf_counter()
    for [nl_str,url] in tqdm(nl_str_list):
        res=cache_code_search(model,device,sourceData,searchNum,nl_str,drawImg)
        statictic.append(print_res(printRes,res,url))
    end = time.perf_counter()
    print_count(statictic)
    print_time(start,end,'cache_experiment',len(nl_str_list))

def prune_experiment(model,device,sourceData,searchNum,nl_str_list,deviation,binaryTime):
    statictic=[]
    start =time.perf_counter()
    for [nl_str,url] in tqdm(nl_str_list):
        res=prune_cache_code_search(model,device,sourceData,searchNum,nl_str,deviation,binaryTime,useSearchNum,drawImg)
        statictic.append(print_res(printRes,res,url))
    end = time.perf_counter()
    res_count=print_count(statictic)
    runTime=print_time(start,end,'prune_experiment',len(nl_str_list))
    return res_count,runTime

def pso_experiment(model,device,sourceData,searchNum,nl_str_list,deviation,binaryTime,pso_param):
    statictic=[]
    start =time.perf_counter()
    for [nl_str,url] in tqdm(nl_str_list):
        res=pso_prune_cache_code_search(model,device,sourceData,searchNum,nl_str,deviation,binaryTime,pso_param,useSearchNum)
        statictic.append(print_res(printRes,res,url))
    end = time.perf_counter()
    res_count=print_count(statictic)
    runTime=print_time(start,end,'pso_experiment',len(nl_str_list))
    return res_count,runTime

def batch_experiment(model,device,sourceData,searchNum,nl_str_list,deviation,binaryTime,batchSize):
    statictic=[]
    start =time.perf_counter()
    for [nl_str,url] in tqdm(nl_str_list):
        res=batch_prune_cache_code_search(model,device,sourceData,searchNum,nl_str,deviation,binaryTime,batchSize,useSearchNum)
        statictic.append(print_res(printRes,res,url))
    end = time.perf_counter()
    res_count=print_count(statictic)
    runTime=print_time(start,end,'batch_experiment',len(nl_str_list))
    return res_count,runTime


def prune_param_experiment(model,device,sourceData,searchNum,nl_str_list):
    # 写入文件
    writePath='./src/prune_res/prune_param_experiment.csv'
    write_file=open(writePath,'w',encoding='utf-8',newline='')
    result_writer = csv.writer(write_file)
    header=['test_num','deviation','binaryTime','runTime','avg_accuracy','avg_similarity_ratio','avg_best_score']
    result_writer.writerow(header)

    # 参数生成
    deviation_list=[0.09]
    binaryTime_list=[30]
    # deviation_list=[]
    # binaryTime_list=[]
    # deviation_range=np.arange(0.09,0.12,0.01)
    # binaryTime_range=np.arange(10,61,10)
    # print(f'******prune_param_num:{len(deviation_range)*len(binaryTime_range)}******')
    # for deviation in deviation_range:
    #     for binaryTime in binaryTime_range:
    #         deviation_list.append(deviation)
    #         binaryTime_list.append(binaryTime)

    # 实验
    for i in tqdm(range(0,len(deviation_list))):
        deviation=deviation_list[i]
        binaryTime=binaryTime_list[i]
        print(f'$$$$$ deviation:{deviation},binaryTime:{binaryTime} $$$$$')
        res_count,runTime=prune_experiment(model,device,sourceData,searchNum,nl_str_list,deviation,binaryTime)
        # 写入文件
        row=[res_count['test_num'],deviation,binaryTime,runTime,res_count['avg_accuracy'],res_count['avg_similarity_ratio'],res_count['avg_best_score']]
        result_writer.writerow(row)

def pso_param_experiment(model,device,sourceData,searchNum,nl_str_list):
    # 写入文件
    writePath='./src/pso_res/pso_param_experiment.csv'
    write_file=open(writePath,'w',encoding='utf-8',newline='')
    result_writer = csv.writer(write_file)
    header=['test_num','c1','c2','w','n_particles','iters','runTime','avg_accuracy','avg_similarity_ratio','avg_best_score']
    result_writer.writerow(header)

    pso_param = { "options":{'c1': 0.5, 'c2': 0.8, 'w': 1.5}  ,"n_particles":20 , "iters":50 ,'n_processes':1}
    # 参数生成
    c1_list=[]
    c2_list=[]
    w_list=[]
    n_particles_list=[]
    iters_list=[]
    c1_range=np.arange(1.6,2.1,0.4)
    c2_range=np.arange(1.6,2.1,0.4)
    w_range=np.arange(0.9,2.0,0.6)
    n_particles_range=np.arange(100,201,100)
    iters_range=np.arange(50,101,25)
    print(f'******pso_param_num:{len(c1_range)*len(c2_range)*len(w_range)*len(n_particles_range)*len(iters_range)}******')
    for c1 in c1_range:
        for c2 in c2_range:
            for w in w_range:
                for n_particles in n_particles_range:
                    for iters in iters_range:
                        c1_list.append(c1)
                        c2_list.append(c2)
                        w_list.append(w)
                        n_particles_list.append(n_particles)
                        iters_list.append(iters)

    # 实验
    for i in tqdm(range(0,len(c1_list))):
        pso_param={ "options":{'c1': c1_list[i], 'c2': c2_list[i], 'w': w_list[i]}  ,"n_particles":n_particles_list[i] , "iters":iters_list[i] ,'n_processes':1}
        print(f'$$$$$ c1:{c1_list[i]},c2:{c2_list[i]},w:{w_list[i]},n_particles:{n_particles_list[i]},iters:{iters_list[i]} $$$$$')
        res_count,runTime=pso_experiment(model,device,sourceData,searchNum,nl_str_list,deviation,binaryTime,pso_param)
        # 写入文件
        row=[res_count['test_num'],c1_list[i],c2_list[i],w_list[i],n_particles_list[i],iters_list[i],runTime,res_count['avg_accuracy'],res_count['avg_similarity_ratio'],res_count['avg_best_score']]
        result_writer.writerow(row)

def batch_param_experiment(model,device,sourceData,searchNum,nl_str_list):
    # 写入文件
    writePath='./src/batch_res/batch_param_experiment.csv'
    write_file=open(writePath,'w',encoding='utf-8',newline='')
    result_writer = csv.writer(write_file)
    header=['test_num','batch_size','runTime','avg_accuracy','avg_similarity_ratio','avg_best_score']
    result_writer.writerow(header)

    # 参数生成
    batch_size_list=[20,40,60,80,100,300,500,800,1000,2500,5000,10000]
    # batch_size_range=np.arange(20,200,20)
    # for batch_size in batch_size_range:
    #     batch_size_list.append(batch_size)
    print(f'******batch_param_num:{len(batch_size_list)}******')

    # 实验
    for i in tqdm(range(0,len(batch_size_list))):
        batch_size=batch_size_list[i]
        print(f'$$$$$ batch_size:{batch_size} $$$$$')
        res_count,runTime=batch_experiment(model,device,sourceData,searchNum,nl_str_list,deviation,binaryTime,batch_size)
        # 写入文件
        row=[res_count['test_num'],batch_size,runTime,res_count['avg_accuracy'],res_count['avg_similarity_ratio'],res_count['avg_best_score']]
        result_writer.writerow(row)


if __name__ == "__main__":
    print("cuda is available : {}".format(torch.cuda.is_available()))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_unixcoder = init_unixcoder(device,"microsoft/unixcoder-base-nine")
    print("init finish")
    language = "python"

    testStart=0
    testNum=1
    nl_str_list = get_nl_list(language,testNum,testStart)
    # nl_str_list = ["Returns an array of bounding boxes of human faces in a image","Adds properties for all fields in this protocol message type.","Make sure a DB specifier exists, creating it if necessary.","Return the datetime truncated to the precision of the provided unit.","Return True if the class is a date type."]
    # nl_str_list = ["Make sure a DB specifier exists, creating it if necessary."]

    searchNum = 50000
    useSearchNum=False

    deviation=100
    binaryTime=3
    batchSize=1000

    drawImg=True
    printRes=False
    isAvg=True

    pso_param = { "options":{'c1': 0.5, 'c2': 0.8, 'w': 1.5}  ,"n_particles":20 , "iters":50 ,'n_processes':1}

    # 缓存数据
    sourceData=cache_get_source_data(language,searchNum) # 注意，这里二分需要读取全部数据
    print("cache finish")
    # time.sleep(5)

    # origin_experiment(model_unixcoder,device,language,searchNum,nl_str_list)
    # normal_experiment(model_unixcoder,device,language,searchNum,nl_str_list)
    # cache_experiment(model_unixcoder,device,sourceData,searchNum,nl_str_list)
    prune_experiment(model_unixcoder,device,sourceData,searchNum,nl_str_list,deviation,binaryTime)
    # pso_experiment(model_unixcoder,device,sourceData,searchNum,nl_str_list,deviation,binaryTime,pso_param)  # 只有第一条数据
    # batch_experiment(model_unixcoder,device,sourceData,searchNum,nl_str_list,deviation,binaryTime,batchSize)

    # prune_param_experiment(model_unixcoder,device,sourceData,searchNum,nl_str_list)
    # pso_param_experiment(model_unixcoder,device,sourceData,searchNum,nl_str_list)
    # batch_param_experiment(model_unixcoder,device,sourceData,searchNum,nl_str_list)
 