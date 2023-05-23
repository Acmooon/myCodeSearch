from enum import Enum

class search_language(str, Enum):
    python = 'python'
    java = 'java'
    javascript = 'javascript'
    php = 'php'
    ruby = 'ruby'
    go = 'go'
    
class all_language(str, Enum):
    python = 'python'
    java = 'java'
    javascript = 'javascript'
    php = 'php'
    ruby = 'ruby'
    go = 'go'
    # 不能搜索
    c = 'c'
    cpp = 'cpp'
    csharp = 'csharp'

