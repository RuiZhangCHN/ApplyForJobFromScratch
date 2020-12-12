# 整数划分问题
# 将正整数n表示成一系列正整数之和
# n = n1+n2+..+nk，其中n1>=n2>=n3>=...>=nk
#
# 在正整数n的所有不通划分中，将最大加数n1不大于m的划分个数记作q(n,m)
# 可以建立如下的递归关系
# (1) q(n,1)=1, n>=1 ———— 当最大加数不大于1时，只有1种划分，即n=1*n
# (2) q(n,m)=q(n,n), m>=n ———— 最大加数不能大于n，故q(1,n) = 1
# (3) q(n,n)=1+q(n,n-1) ———— 划分由m=n和m=n-1的划分组成
# (4) q(n,m) = q(n,m-1)+q(n-m,m), n>m>1 ———— 正整数的最大加数n1不大于m的划分由n1=m和n1<=m-1的划分构成

def q(n, m):
    if n<1 or m < 1:
        return 0
    elif m == 1 or n == 1:
        return 1
    elif n<m:
        return q(n,n)
    elif n == m:
        return 1+q(n,m-1)
    else:
        return q(n-m, m) + q(n,m-1)

def q_entry(n):
    return q(n,n)

print(q_entry(6))