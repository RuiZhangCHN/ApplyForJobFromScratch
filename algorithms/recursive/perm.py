# 排列问题
# 对R = {r1,r2,r3,...,rn}进行排列，其中Ri = R - {ri}
# perm(X)表示X的全排列
# (ri)perm(X)表示在X全排列的前面加上ri元素
#
# 因此全排列可以归纳定义为
# （1）当n=1时，perm(R) = (r)
# （2）当n>1时，perm(R) = (r1)perm(R1)+(r2)perm(R2)+...+(rn)perm(Rn)

def perm(nums):

    if len(nums) == 1:
        return [nums]

    else:
        ret = []
        for i in range(len(nums)):
            r_i = nums[i]
            R_i = nums[:i] + nums[i+1:]
            for res in perm(R_i):
                ret.append([r_i] + res)

        return ret

print(perm([1,2,3,4]))
print(len(perm([1,2,3,4])))