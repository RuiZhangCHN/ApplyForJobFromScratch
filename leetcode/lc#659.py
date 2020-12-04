
# 659. 分割数组为连续子序列
# 给你一个按升序排序的整数数组 num（可能包含重复数字），请你将它们分割成一个或多个子序列，其中每个子序列都由连续整数组成且长度至少为 3 。
#
# 如果可以完成上述分割，则返回 true ；否则，返回 false 。

import heapq
import collections

class Solution:
    def isPossible(self, nums):

        mydict = {}
        for x in nums:
            if mydict.get(x-1, []):
                queue = mydict.get(x-1)
                prevLength= heapq.heappop(queue)
                if x not in mydict.keys():
                    mydict[x] = []
                heapq.heappush(mydict[x], prevLength+1)
            else:
                if x not in mydict.keys():
                    mydict[x] = []
                heapq.heappush(mydict[x], 1)

            print(mydict)

        return not any(queue and queue[0] < 3 for queue in mydict.values())

# 贪心解法：
# 使用两个哈希表，第一个哈希表存储数组中每个数字的剩余次数，第二个哈希表存储数组中每个数字作为结尾的子序列数量
# (1)首先判断是否存在以x-1结尾子序列，即根据第二个哈希表判断x-1作为结尾的子序列的数量是否大于0.如果大于0，则将x加入该子序列中，
# 因为在这里使用了x，所以在第一个哈希表中减掉x的1次剩余次数。同时在第二个哈希表中x-1作为结尾的子序列减1，x结尾的加1.
# (2)否则，x作为子序列的第一个数，需要看x+1,x+2有没有剩余次数，如果没有就无法分割，返回false
# (3)如果完全遍历完没有无法分割的情况，返回true

class Solution2:
    def isPossible(self, nums):
        countMap = collections.Counter(nums)
        endMap = collections.Counter()

        for x in nums:
            count = countMap[x]
            if count > 0:
                prevEndCount = endMap.get(x-1, 0)
                if prevEndCount > 0:
                    countMap[x] -= 1
                    endMap[x-1] = prevEndCount - 1
                    endMap[x] += 1
                else:
                    count1 = countMap.get(x+1, 0)
                    count2 = countMap.get(x+2, 0)
                    if count1> 0 and count2 > 0:
                        countMap[x] -= 1
                        countMap[x+1] -= 1
                        countMap[x+2] -= 1
                        endMap[x+2] += 1
                    else:
                        return False

        return True

s = Solution2()
print(s.isPossible([1,2,3,3,4,5]))
print(s.isPossible([1,2,3,3,4,5,5]))
print(s.isPossible([1,2,3,3,4,4,5]))
print(s.isPossible([1,2,3,3,4,4,5,5]))