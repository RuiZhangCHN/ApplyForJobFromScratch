
class Solution:

    def selectionSort(self, nums):

        for i in range(len(nums)):
            min_idx = i
            for j in range(i, len(nums)):
                if nums[j] < nums[min_idx]:
                    min_idx = j

            nums[i], nums[min_idx] = nums[min_idx], nums[i]

        return nums


solution = Solution()

# case 1: []
c1 = []
print(solution.selectionSort(c1))

# case 2: [6,3,7,9,2,5,4,1] -> [1, 2, 3, 4, 5, 6, 7, 9]
c2 = [6,3,7,9,2,5,4,1]
print(solution.selectionSort(c2))

# case 3: [8,4,7,6,9,2,7,5,8,3,8] -> [2, 3, 4, 5, 6, 7, 7, 8, 8, 8, 9]
c3 = [8,4,7,6,9,2,7,5,8,3,8]
print(solution.selectionSort(c3))