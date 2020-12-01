
class Solution:

    def shellSort(self, nums):

        n = len(nums)

        # 初始步长
        gap = n // 2
        while(gap > 0):
            for i in range(gap, n):
                temp = nums[i]
                j = i
                while (j >= gap) and nums[j - gap] > temp:
                    nums[j] = nums[j - gap]
                    j -= gap
                nums[j] = temp

            gap = gap // 2

        return nums

solution = Solution()

# case 1: []
c1 = []
print(solution.shellSort(c1))

# case 2: [6,3,7,9,2,5,4,1] -> [1, 2, 3, 4, 5, 6, 7, 9]
c2 = [6,3,7,9,2,5,4,1]
print(solution.shellSort(c2))

# case 3: [8,4,7,6,9,2,7,5,8,3,8] -> [2, 3, 4, 5, 6, 7, 7, 8, 8, 8, 9]
c3 = [8,4,7,6,9,2,7,5,8,3,8]
print(solution.shellSort(c3))