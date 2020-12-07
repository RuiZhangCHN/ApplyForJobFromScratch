class Solution:

    def interpolation_search(self, list, target):
        low = 0
        high = len(list) - 1

        while(low < high):
            mid = low + int((target-list[low])/(list[high]-list[low])*(high-low))
            if target < list[mid]:
                high = mid-1
            elif target > list[mid]:
                low = mid+1
            else:
                return mid
        if low == high and list[mid] == target:
            return True

        return False

l = [2,5,7,9,15,18,26,29,33,35,42,47,55,56,60,66,68,71,72,73,74,77,80,81,83,84,86,88,89,90,91,92,93,95,97,101]

solution = Solution()
print(solution.interpolation_search(l, 2))
print(solution.interpolation_search(l, 5))
print(solution.interpolation_search(l, 88))
print(solution.interpolation_search(l, 90))
print(solution.interpolation_search(l, 101))
print(solution.interpolation_search(l, 102))
print(solution.interpolation_search(l, 63))
print(solution.interpolation_search(l, 27))
print(solution.interpolation_search(l, 67))
print(solution.interpolation_search(l, 43))