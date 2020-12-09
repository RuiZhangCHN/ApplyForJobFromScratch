
class Entity:
    def __init__(self, key, value):
        self.key = key
        self.value = value


class Node:
    def __init__(self):
        self.parent = None
        self.entities = []
        self.children = []

    def find(self, key):
        for e in self.entities:
            if key == e.key:
                return e

        return None

    def delete(self, key):
        for i, e in enumerate(self.entities):
            if e.key == key:
                del self.entities[i]
                return (i, e)

    def isLeaf(self):
        return len(self.children) == 0

    def addEntity(self, entity):
        self.entities.append(entity)
        self.entities.sort(key=lambda x: x.key)

    def addChild(self, node):
        self.children.append(node)
        node.parent = self
        self.children.sort(key=lambda x: x.entities[0].key)

class Tree:

    def __init__(self, size=6):
        self.size = size
        self.root = None
        self.length = 0

    def add(self, key, value=None):

        self.length += 1

        if self.root:
            current = self.root

            while not current.isLeaf():
                for i, e in enumerate(current.entities):
                    if e.key > key:
                        current = current.childs[i]
                        break
                    elif e.key == key:
                        e.val = value
                        self.length -= 1
                        return
