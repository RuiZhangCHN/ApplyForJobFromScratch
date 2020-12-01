
# 【对称的二叉树】
# 三个评估条件：(1) L.val == R.val (2) L.R == R.L (3) L.L == R.R

class Solution:

    def compare(self, left, right):
        if not left and not right:
            return True
        if not left or not right:
            return False
        return left.val == right.val and self.compare(left.right, right.left) and self.compare(left.left, right.right)


    def isSymmetric(self, root):
        if not root:
            return True
        return self.compare(root.left, root.right)