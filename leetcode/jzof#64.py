# 剑指 Offer 64. 求1+2+…+n
# 求 1+2+...+n ，要求不能使用乘除法、for、while、if、else、switch、case等关键字及条件判断语句（A?B:C）。

# 法一：等差数列求和
# S_n = n*a_1 + n*(n-1)/2*d

import math

class Solution:
    def sumNums(self, n: int) -> int:
        return (int(math.pow(n,2)+n)>>1)

# 法二：递归
class Solution:
    def sumNums(self, n: int) -> int:
        return n and n + self.sumNums(n-1)