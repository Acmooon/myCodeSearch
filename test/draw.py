import matplotlib.pyplot as plt
from datetime import datetime

def drawSimpleImg(data,name='test'):
    # plt.clf()
    now = datetime.now()
    filename = now.strftime(f'%m%d_%H-%M-%S') + '.png'
    save_file=f'./src/Img/{name}_{filename}'

    plt.rcParams['lines.markersize'] = 2
    plt.xscale('linear')
    plt.plot(data, marker='o')
    plt.show()
    # plt.gcf().set_size_inches(100, 10)
    # plt.savefig(save_file,dpi=100, bbox_inches=None, pad_inches=0.1)

def drawDoubledata(data,data2,name='test'):
    # 设置全局字体大小为14
    plt.rcParams.update({'font.size': 20})

    # plt.clf()
    # now = datetime.now()
    # filename = now.strftime(f'%m%d_%H-%M-%S') + '.png'
    # save_file=f'./src/Img/{name}_{filename}'

    plt.rcParams['lines.markersize'] = 2
    plt.xscale('linear')
    plt.plot(data, marker='o')
    plt.plot(data2, marker='o')
    plt.show()
    # plt.gcf().set_size_inches(100, 10)
    # plt.savefig(save_file,dpi=100, bbox_inches=None, pad_inches=0.1)

if __name__=='__main__':
    data = [0, 1, 2, 3, 3, 2, 1, 0, 1, 1]   
    # drawSimpleImg(data,"test")
    data2=range(0, 10)
    data=data+list(data2)
    drawSimpleImg(data,"test2")