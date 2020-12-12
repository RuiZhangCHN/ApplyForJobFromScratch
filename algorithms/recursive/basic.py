# 阶乘函数
def factorial(n):
    if n == 0:
        return 1
    else:
        return n * factorial(n-1)

# 斐波那契数列
def fibonacci(n):
    if n <= 1:
        return 1
    else:
        return fibonacci(n-1) + fibonacci(n-2)

# Ackerman函数
# 当一个函数以及它的一个变量是由函数自身定义时，称之为双递归函数。
def Ackerman(n, m):
    if n == 1 and m == 0:
        return 2
    elif n == 0:    # and m >=0
        return 1
    elif m == 0:    # and n >=2
        return n+2
    else:
        return Ackerman(Ackerman(n-1, m), m-1)

