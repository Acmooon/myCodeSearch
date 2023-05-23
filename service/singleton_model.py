import sys
sys.path.append("./model")
from unixcoder import init_unixcoder


class Singleton_model:
    def __init__(self,device='cpu'):
        self.model_unixcoder = init_unixcoder(device,"microsoft/unixcoder-base-nine")

    def get_model(self,model_name):
        if(model_name=="unixcoder"):
            return self.model_unixcoder
        else:
            return None