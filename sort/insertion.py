class Solution:

    def insertionSort(self, nums):

        sorted_idx = 0

        for i in range(1, len(nums)):
            pre_idx = i - 1
            current_num = nums[i]

            while pre_idx >=0 and nums[pre_idx] > current_num:
                nums[pre_idx+1] = nums[pre_idx]
                pre_idx -= 1

            nums[pre_idx+1] = current_num

        return nums

solution = Solution()

# case 1: []
c1 = []
print(solution.insertionSort(c1))

# case 2: [6,3,7,9,2,5,4,1] -> [1, 2, 3, 4, 5, 6, 7, 9]
c2 = [6,3,7,9,2,5,4,1]
print(solution.insertionSort(c2))

# case 3: [8,4,7,6,9,2,7,5,8,3,8] -> [2, 3, 4, 5, 6, 7, 7, 8, 8, 8, 9]
c3 = [8,4,7,6,9,2,7,5,8,3,8]
print(solution.insertionSort(c3))