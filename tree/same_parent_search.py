
# 【二叉搜索树的最近公共祖先 - 后序遍历DFS】
# 注：可以利用值比较加快一下速度
# 1. 终止条件：当到达叶子节点，直接返回null;当root等于p或者q，直接返回root
# 2. 比较值，如果p,q都比root大，搜索右子树；如果都比root小，搜索左子树；一大一小说明应返回root

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

        if root.val > p and root.val > q:
            return self.lowestCommonAncestor(root.left, p, q)
        elif root.val < p and root.val < q:
            return self.lowestCommonAncestor(root.right, p, q)
        else:
            return root


testcase = TreeNode(6, left=TreeNode(2,
                                     left=TreeNode(0),
                                     right=TreeNode(4,
                                                    left=TreeNode(3),
                                                    right=TreeNode(5))),
                    right=TreeNode(8,
                                   left=TreeNode(7),
                                   right=TreeNode(9)))

solution = Solution()
ret = solution.lowestCommonAncestor(testcase, 2, 8)

print(ret.val)