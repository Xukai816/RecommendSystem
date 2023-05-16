import os


def mkdirs(path):
    if not os.path.exists(path):
        os.makedirs(path)


def params_count(model):
    print('#-------------------#')
    params = list(model.parameters())
    k = 0
    for i in params:
        l = 1
        print("该层的结构：" + str(list(i.size())))
        for j in i.size():
            l *= j
        print("该层参数和：" + str(l))
        k = k + l
    print("总参数数量和：" + str(k))
    print('#-------------------#')

