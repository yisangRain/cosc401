class DTNode:
    def __init__(self, decision):
        # either a function or a value
        self.decision = decision
        self.children = None


    def predict(self, inputObject):
        if self.children == None:
            return self.decision
        else:
            return self.children[self.decision(inputObject)].predict(inputObject)
        

    def leaves(self):
        if self.children == None:
            return 1
        else:
            leaf_number = 0
            for child in self.children:
                leaf_number += child.leaves()
            return leaf_number
            


def q1_test():
    # The following (leaf) node will always predict True
    node = DTNode(True) 

    # Prediction for the input (1, 2, 3):
    x = (1, 2, 3)
    print(node.predict(x))

    # Sine it's a leaf node, the input can be anything. It's simply ignored.
    print(node.predict(None))
    # True
    # True


    yes_node = DTNode("Yes")
    no_node = DTNode("No")
    tree_root = DTNode(lambda x: 0 if x[2] < 4 else 1)
    tree_root.children = [yes_node, no_node]

    print(tree_root.predict((False, 'Red', 3.5)))
    print(tree_root.predict((False, 'Green', 6.1)))
    # Yes
    # No

def q2_test():
    n = DTNode(True)
    print(n.leaves())
    # 1

    t = DTNode(True)
    f = DTNode(False)
    n = DTNode(lambda v: 0 if not v else 1)
    n.children = [t, f]
    print(n.leaves())
    # 2

# q1_test()
q2_test()