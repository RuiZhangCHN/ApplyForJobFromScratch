
class Solution:

    def bubbleSort(self, nums):

        for i in range(1, len(nums)):
            for j in range(len(nums)-i):
                if nums[j] > nums[j+1]:
                    nums[j+1], nums[j] = nums[j], nums[j+1]

        return nums

solution = Solution()

# case 1: []
c1 = []
print(solution.bubbleSort(c1))

# case 2: [6,3,7,9,2,5,4,1] -> [1, 2, 3, 4, 5, 6, 7, 9]
c2 = [6,3,7,9,2,5,4,1]
print(solution.bubbleSort(c2))

# case 3: [8,4,7,6,9,2,7,5,8,3,8] -> [2, 3, 4, 5, 6, 7, 7, 8, 8, 8, 9]
c3 = [8,4,7,6,9,2,7,5,8,3,8]
print(solution.bubbleSort(c3))