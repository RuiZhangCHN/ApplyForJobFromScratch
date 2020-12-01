
class Solution:

    def quickSort(self, nums):
        if len(nums) <= 1:
            return nums

        pivot = nums[0]
        less = [item for item in nums[1:] if item < pivot]
        greater = [item for item in nums[1:] if item >= pivot]
        return self.quickSort(less) + [pivot] + self.quickSort(greater)


solution = Solution()

# case 1: []
c1 = []
print(solution.quickSort(c1))

# case 2: [6,3,7,9,2,5,4,1] -> [1, 2, 3, 4, 5, 6, 7, 9]
c2 = [6,3,7,9,2,5,4,1]
print(solution.quickSort(c2))

# case 3: [8,4,7,6,9,2,7,5,8,3,8] -> [2, 3, 4, 5, 6, 7, 7, 8, 8, 8, 9]
c3 = [8,4,7,6,9,2,7,5,8,3,8]
print(solution.quickSort(c3))