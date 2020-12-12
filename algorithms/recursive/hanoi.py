# 汉诺塔问题
# n表示塔座上的圆盘的数量，要将其从a移到b
# 当n=1时，只需要直接把它从a移到b
# 当n>1时，需要使用辅助塔c，先将n-1块移动到c，再移动第n块

def move(a,b):
    pass

def hanoi(n,a,b,c):
    if n > 0:
        hanoi(n-1, a, c, b)
        move(a,b)
        hanoi(n-1, c, b, a)

