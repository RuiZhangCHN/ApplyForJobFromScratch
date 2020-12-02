
# 【根据前序遍历和中序遍历重构二叉树】
# 核心要点 找到parent节点，切分出左右子树的前序和中序片段，递归即可

class TreeNode:
    def __init__(self, x):
        self.val = x
        self.left = None
        self.right = None

class Solution:
    def buildTree(self, preorder, inorder):

        if len(preorder) == 0:
            return None

        # 获取第一个节点
        root_node = TreeNode(preorder[0])

        # 获取左子树
        for i in range(len(inorder)):
            if inorder[i] == preorder[0]:
                break

        # 左子树
        left_inorder = inorder[:i]
        left_preorder = preorder[1:i+1]

        # 右子树
        right_inorder = inorder[i+1:]
        right_preorder = preorder[i+1:]

        root_node.left = self.buildTree(left_preorder, left_inorder)
        root_node.right = self.buildTree(right_preorder, right_inorder)

        return root_node