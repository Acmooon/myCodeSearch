from app.selfmodel.bo_zh import main_t
from cnocr import CnOcr  # type:ignore
from cnstd import CnStd  # type:ignore yum install mesa-libGL -y
import multiprocessing
from multiprocessing import managers
from concurrent.futures import ProcessPoolExecutor
from app.ocr_cn import *
from app.selftranslation import *
from app.translation import *
from utils.tools import *
from setting.setting import *


class GlobalObject:
    def __init__(self) -> None:
        self.ocr_std = CnStd()
        self.model_dict = self.create_model_dict()

    def create_model_dict(self):
        model_dict = {"Helsinki-NLP/opus-mt-en-zh": get_var("utils/save_var/opus_mt_en_zh.bin")} # 这是被序列化之后保存在本地的模型
        return model_dict

    def getStd(self):
        return self.ocr_std

    def get_model(self, model_list):
        models = []
        for i in model_list:
            models.append(self.model_dict.get(i))
        return models

    def ocr(self, path):
        res_cn, soc_cn = ocr_std(path, self) # 遇到不能够序列化的时候可以传递整个对象到到函数当中去，就可以避免这个问题了
        return res_std, soc_std


class MyManager(managers.BaseManager):
    pass# 自定义多进程类


res_dict = {}


def proc_callback(res):
    res_dict[res.result()['task_id']] = res.result()['res']
    return res_dict # 需要一个全局变量用于捕捉结果


def proc_worker_ocr(gobj, task_id, path: str = ""):
    return {"task_id": task_id, "res": gobj.ocr(path)}


def proc_worker_standTran(gobj, task_id, tran_dict):
    models = gobj.get_model(tran_dict.get("model_name_list"))
    return {"task_id": task_id, "res": tran_distribution(models, tran_dict)}


def proc_worker_default(task_id):
    return task_id


class ServerExecutor:
    def __init__(self):
        # 在Manager中注册自定义类（GlobalObject是我的自定义类, 类内部分别包含普通模型与复杂模型两种）
        MyManager.register("GlobalObject", GlobalObject)
        manager = MyManager()
        manager.start()
        # 创建共享对象
        self.global_object = manager.GlobalObject()  # type: ignore
        # 这里不仅可以是ProcessPoolExecutor，也可以是多进程Process或者进程池Pool，各自用法略有不同
        _cpu_cunt = multiprocessing.cpu_count() if cpu_cunt == None else cpu_cunt # 配置文件获取两个进程池占用比例以及CPU总数信息
        self.executor = ProcessPoolExecutor(round(_cpu_cunt*ocr_cunt)) # 这里可以开展多个进程池，进程池不会互相干扰
        self.executor_trans = ProcessPoolExecutor(round(_cpu_cunt*tran_cunt))

    def submit(self, task_id, task_type, **kwargs):
        if(task_type == "OcrManager"):
            future = self.executor.submit(
                proc_worker_ocr, self.global_object, task_id, path=kwargs.get("path", ""))
        elif(task_type == "TranManager"):
            future = self.executor_trans.submit(
                proc_worker_standTran, self.global_object, task_id, tran_dict=kwargs.get("tran_dict"))
        else:
            future = self.executor.submit(proc_worker_default, task_id)
        return future
    
executor = ServerExecutor()