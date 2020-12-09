

class Node:
    def __init__(self, val, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

class BinSearchTree:
    def __init__(self):
        self.root = None

    def add(self, x):

        node = Node(x)

        if self.root is None:
            self.root = node
            return
        else:
            y = self.root

            while True:
                if y.val > node.val:
                    if y.left == None:
                        y.left = node
                        return
                    else:
                        y = y.left
                else:
                    if y.right == None:
                        y.right = node
                        return
                    else:
                        y = y.right
