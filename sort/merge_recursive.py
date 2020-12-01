
class Solution:

    def merge(self, lefts, rights):
        ret = []
        lp, rp = 0, 0

        while lp < len(lefts) and rp < len(rights):
            if lefts[lp] < rights[rp]:
                ret.append(lefts[lp])
                lp += 1
            else:
                ret.append(rights[rp])
                rp += 1

        if lp < len(lefts):
            ret = ret + lefts[lp:]
        elif rp < len(rights):
            ret = ret + rights[rp:]

        return ret

    def mergeSort(self, nums):

        if len(nums) <= 1:
            return nums

        mid = len(nums) // 2
        left = self.mergeSort(nums[:mid])
        right = self.mergeSort(nums[mid:])
        return self.merge(left, right)



solution = Solution()

# case 1: []
c1 = []
print(solution.mergeSort(c1))

# case 2: [6,3,7,9,2,5,4,1] -> [1, 2, 3, 4, 5, 6, 7, 9]
c2 = [6,3,7,9,2,5,4,1]
print(solution.mergeSort(c2))

# case 3: [8,4,7,6,9,2,7,5,8,3,8] -> [2, 3, 4, 5, 6, 7, 7, 8, 8, 8, 9]
c3 = [8,4,7,6,9,2,7,5,8,3,8]
print(solution.mergeSort(c3))