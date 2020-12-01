
# 【最近公共祖先 - 后序遍历DFS】
# 1. 终止条件：当到达叶子节点，直接返回null;当root等于p或者q，直接返回root
# 2. 递归左子节点，记为left；递归右子节点，记为right
# 3. (1) 当left和right都为空，说明root的左右子树不包含p,q，返回Null
#    (2) 当left和right都不为空，说明p,q分别在root的左侧和右侧，返回root
#    (3) 当left空right不空，则直接返回right；当left不空right空，则直接返回left

class TreeNode:
    def __init__(self, x, left=None, right=None):
        self.val = x
        self.left = left
        self.right = right

class Solution:
    def lowestCommonAncestor(self, root, p, q):

        if not root:
            return None

        if root.val == p or root.val == q:
            return root

        left = self.lowestCommonAncestor(root.left, p, q)
        right = self.lowestCommonAncestor(root.right, p, q)

        if left and right:
            return root

        elif left or right:
            if left:
                return left
            else:
                return right
        else:
            return None


testcase = TreeNode(3, left=TreeNode(5,
                                     left=TreeNode(6),
                                     right=TreeNode(2,
                                                    left=TreeNode(7),
                                                    right=TreeNode(4))),
                    right=TreeNode(1,
                                   left=TreeNode(0),
                                   right=TreeNode(8)))

solution = Solution()
ret = solution.lowestCommonAncestor(testcase, 5, 4)

print(ret.val)