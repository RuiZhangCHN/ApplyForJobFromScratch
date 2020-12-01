
# 【二叉树的深度】
# 等于max(左子树的最大深度, 右子树的最大深度) + 1

class Solution:
    def maxDepth(self, root):

        if not root:
            return 0

        return max(self.maxDepth(root.left), self.maxDepth(root.right)) + 1