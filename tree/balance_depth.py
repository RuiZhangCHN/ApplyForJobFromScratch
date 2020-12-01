
# 【平衡二叉树的深度】
# 如果某二叉树中任意节点的左右子树的深度相差不超过1，那么它就是一棵平衡二叉树。
# 判断其左右子树的深度差，然后递归处理子节点

class Solution:

    def maxDepth(self, root):
        if not root:
            return 0

        return max(self.maxDepth(root.left), self.maxDepth(root.right)) + 1

    def isBalanced(self, root: TreeNode) -> bool:

        if not root:
            return True

        left_depth = self.maxDepth(root.left)
        right_depth = self.maxDepth(root.right)

        if left_depth - right_depth < -1 or left_depth - right_depth > 1:
            return False
        else:
            return True and self.isBalanced(root.left) and self.isBalanced(root.right)