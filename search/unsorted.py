
class Solution:

    def sequential_search(self, list, target):
        length = len(list)

        for idx in range(length):
            if list[idx] == target:
                return idx

        return None


solution = Solution()

print(solution.sequential_search([6,2,7,4,8,1], 7))
print(solution.sequential_search([6,2,7,4,8,1], 5))