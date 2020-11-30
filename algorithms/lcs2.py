
# 定义：res[i][j]定义为截止字符串A的第i个字符和截止到字符串B的第j个字符的最长公共子串
# 当 i=0, j=0时，res[i][j] = 0
# 当 A[i] == B[j], res[i][j] = res[i-1][j-1] + 1
# 当 A[i] != B[j], res[i][j] = 0
# 下面这个代码要注意考虑空串的情况

def LCS2(stringA, stringB):
    ret = [[0 for _ in range(len(stringB)+1)] for _ in range(len(stringA)+1)]
    max_same = 0
    max_cs = ''
    for i in range(1, len(stringA)+1):
        for j in range(1, len(stringB)+1):
            if stringA[i-1] == stringB[j-1]:
                ret[i][j] = ret[i-1][j-1]+1
                if ret[i][j] > max_same:
                    max_same=ret[i][j]
                    max_cs = stringB[j-max_same:j]

    return max_same, max_cs

print(LCS2("helloworld","loop"))