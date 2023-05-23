from service.code_search import cache_get_source_data


class Cache_data:
    def __init__(self,searchNum=55000):
        self.python_data=cache_get_source_data("python",searchNum)

    def get_data(self,language):
        if(language=="python"):
            return self.python_data
        else:
            return None
