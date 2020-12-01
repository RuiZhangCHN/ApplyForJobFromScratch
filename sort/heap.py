
class Solution:

    def pop_up(self, myheap, cur_idx):
        parent_idx = (cur_idx-1) >> 1
        while (cur_idx) and myheap[parent_idx] > myheap[cur_idx]:
            myheap[cur_idx], myheap[parent_idx] = myheap[parent_idx], myheap[cur_idx]
            cur_idx = parent_idx
            parent_idx = (cur_idx-1) >> 1

        return myheap

    def heapSort(self, nums):

        my_heap = []

        for item in nums:
            my_heap.append(item)
            my_heap = self.pop_up(my_heap, len(my_heap)-1)

        return my_heap



solution = Solution()

# case 1: []
c1 = []
print(solution.heapSort(c1))

# case 2: [6,3,7,9,2,5,4,1] -> [1, 2, 3, 4, 5, 6, 7, 9]
c2 = [6,3,7,9,2,5,4,1]
print(solution.heapSort(c2))

# case 3: [8,4,7,6,9,2,7,5,8,3,8] -> [2, 3, 4, 5, 6, 7, 7, 8, 8, 8, 9]
c3 = [8,4,7,6,9,2,7,5,8,3,8]
print(solution.heapSort(c3))