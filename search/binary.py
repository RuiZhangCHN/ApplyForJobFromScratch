

class Solution:

    def binary_search(self, list, target):
        low = 0
        high = len(list) - 1

        while(low <= high):
            mid = (low+high) // 2
            if target < list[mid]:
                high = mid-1
            elif target > list[mid]:
                low = mid+1
            else:
                return mid

        return False

l = [2,5,7,9,15,18,26,29,33,35,42,47,55,56,60,66,68,71,72,73,74,77,80,81,83,84,86,88,89,90,91,92,93,95,97,101]

solution = Solution()
print(solution.binary_search(l, 15))
print(solution.binary_search(l, 55))
print(solution.binary_search(l, 88))
print(solution.binary_search(l, 90))
print(solution.binary_search(l, 101))
print(solution.binary_search(l, 102))
print(solution.binary_search(l, 63))
print(solution.binary_search(l, 27))
print(solution.binary_search(l, 67))
print(solution.binary_search(l, 43))