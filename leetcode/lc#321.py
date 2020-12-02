# 321. 拼接最大数
# 给定长度分别为 m 和 n 的两个数组，其元素由 0-9 构成，表示两个自然数各位上的数字。现在从这两个数组中选出 k (k <= m + n) 个数字拼接成一个新的数，
# 要求从同一个数组中取出的数字保持其在原数组中的相对顺序。
#
# 求满足该条件的最大数。结果返回一个表示该最大数的长度为 k 的数组。

# 基本思路是对于每种枚举情况构建一个单调栈，然后merge
# 注意merge函数里面的神奇比较方法（max每次比较两个list的第一个元素，返回这个大的list，然后pop它，就可以少掉很多代码）

class Solution:
    def maxNumber(self, nums1, nums2, k):

        def getMaxArr(nums, i):
            if not i:
                return []

            # pop: 最多可以抛弃多少个数字
            stack, pop = [], len(nums)-i
            for num in nums:
                while pop and stack and stack[-1] < num:
                    pop -= 1
                    stack.pop()
                stack.append(num)

            return stack[:i]

        def merge(tmp1, tmp2, k):
            # ????这是什么神仙写法
            return [max(tmp1, tmp2).pop(0) for _ in range(k)]

        res = [0 for _ in range(k)]
        for i in range(k+1):
            if i <= len(nums1) and k - i <= len(nums2):
                tmp1 = getMaxArr(nums1, i)
                tmp2 = getMaxArr(nums2, k-i)

                tmp = merge(tmp1, tmp2, k)
                if res < tmp:
                    res = tmp

        return res