import copy

def fibonacci_sequence(num: int):  # 按照待查找数列的大小，动态生成斐波那契数列
    a, b = 0, 1
    while a <= num-1:
        yield a
        a, b = b, a + b
    yield a
    return

class Solution:

    def fibonacci_search(self, list, target, max_size):

        D = copy.deepcopy(list)

        F = [i for i in fibonacci_sequence(max_size)]

        index = 0
        while(F[index] < len(D)):
            index += 1

        for i in range(len(D), F[index]):
            D.append(D[-1])

        left = 0
        right = F[index]
        while left <=right and index>0:
            mid = left + F[index-1] - 1

            if D[mid] == target:
                if mid > len(list):
                    return len(list) - 1
                else:
                    return mid

            elif D[mid] < target:
                left = mid + 1
                index = index-2
            elif D[mid] > target:
                right = mid-1
                index = index-1

        return None

l = [2,5,7,9,15,18,26,29,33,35,42,47,55,56,60,66,68,71,72,73,74,77,80,81,83,84,86,88,89,90,91,92,93,95,97,101]

s = Solution()

print(s.fibonacci_search(l, 2, 100))
print(s.fibonacci_search(l, 5, 100))
print(s.fibonacci_search(l, 88, 100))
print(s.fibonacci_search(l, 90, 100))
print(s.fibonacci_search(l, 101, 100))
print(s.fibonacci_search(l, 102, 100))
print(s.fibonacci_search(l, 63, 100))
print(s.fibonacci_search(l, 27, 100))
print(s.fibonacci_search(l, 67, 100))
print(s.fibonacci_search(l, 43, 100))