
# 【最大子序和】
# 使用f(i)表示以第i个数结尾的连续子数组的最大和
# 动态规划方程：f(i) = max{f(i-1)+a_i, a_i}

class Solution:
    def maxSubArray(self, nums: List[int]) -> int:
        max_ans = nums[0]
        pre = 0

        for i in range(len(nums)):
            pre = max(pre+nums[i], nums[i])
            max_ans = max(pre, max_ans)

        return max_ans