# 数组中数字出现的次数 II
# 在一个数组 nums 中除一个数字只出现一次之外，其他数字都出现了三次。请找出那个只出现一次的数字。

# 出现了3次，考虑使用异或运算

# 基本思路：
# num = 234
# one, two = 0, 0
# one = one^num & ~two
# two = two^num & ~one
# print(one, two)
# one = one^num & ~two
# two = two^num & ~one
# print(one, two)
# one = one^num & ~two
# two = two^num & ~one
# print(one, two)
# one = one^num & ~two
# two = two^num & ~one
# print(one, two)
# one = one^num & ~two
# two = two^num & ~one
# print(one, two)
# one = one^num & ~two
# two = two^num & ~one
# print(one, two)

class Solution:
    def singleNumber(self, nums: List[int]) -> int:
        ones, twos = 0, 0
        for num in nums:
            ones = ones ^ num & ~twos
            twos = twos ^ num & ~ones
        return ones