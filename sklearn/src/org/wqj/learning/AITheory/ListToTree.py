import json


# 新建一个中间类  后端需要关注
class TreeNode:

    def __init__(self, id, content, parentid, children, inserttime):
        self.id = id
        self.content = content
        self.parentid = parentid
        self.children = children
        self.inserttime = inserttime

    def __str__(self):
        childidarr = [str(item.id) for item in self.children]
        strids = ",".join(childidarr)
        return 'id：%d  内容：%s  父id：%d  子ids : %s  插入时间:%d' % (self.id, self.content, self.parentid, strids, self.inserttime)

    def __repr__(self):
        return repr((self.id, self.content, self.parentid, self.children, self.inserttime))


# list转为树  后端需要关注
def ListToTree(nodelist, parentcode):
    nodetree = []
    for item in nodelist:
        if (item.parentid == parentcode):
            # 传递的是地址
            nodetree.append(item)
        for item2 in nodelist:
            if (item2.parentid == item.id):
                # 利用地址在原来的内存上进行新增
                item.children.append(item2)
    return nodetree


# 前序遍历  前端需要关注
def preOrder(tree):
    if (len(tree) == 0):
        print("为空")
        return
    # 新建一个栈
    stack = []
    top = -1

    # 记住层数,第一层是对消息回复,第二层是对用户id回复
    nodelevelstack = []
    nodePUserAccountStack = []
    # 先将外面的两个node压入栈中,
    # 也就是时间越小的 越靠右边的子树,但是却最先入栈
    for item in reversed(tree):
        top = top + 1
        nodelevelstack.insert(top, 1)
        nodePUserAccountStack.insert(top, "根")
        stack.insert(top, item)

    while (top > -1):
        # 出栈
        tempnode = stack[top]
        nodelevel = nodelevelstack[top]
        PUseraccount = nodePUserAccountStack[top]
        # 这里读取节点,可以将数据存在一个新的list中
        # 输出,在显示中第一层的直接对评论回复,第二层以上的 对用户id回复,可以取PUseraccount值
        print("节点信息:%s     ++++++层级:%d    父级内容:%s" % (str(tempnode), nodelevel, PUseraccount))
        top = top - 1

        # 将他的子节点压入栈中,也需要保证排序,时间小的先进去
        for child in reversed(tempnode.children):
            top = top + 1
            # 子节点比父节点层数要高一级
            nodelevelstack.insert(top, nodelevel + 1)
            nodePUserAccountStack.insert(top, tempnode.content)
            stack.insert(top, child)


if __name__ == '__main__':
    nodelist = []
    nodelist.append(TreeNode(1, 'content1', -1, [], 1))
    nodelist.append(TreeNode(2, 'content2', -1, [], 2))
    nodelist.append(TreeNode(3, 'content3', 1, [], 3))
    nodelist.append(TreeNode(4, 'content4', 1, [], 4))
    nodelist.append(TreeNode(5, 'content5', 3, [], 5))
    nodelist.append(TreeNode(6, 'content6', 5, [], 6))
    nodelist.append(TreeNode(7, 'content7', 3, [], 7))
    nodelist.append(TreeNode(8, 'content8', 2, [], 8))
    # 按照插入时间进行排序,按照逆序,
    sortnodelist = nodelist.sort(key=lambda item: item.inserttime, reverse=False)
    # list转tree
    tree = ListToTree(nodelist, -1)

    print(json.dumps(tree, default=lambda item: item.__dict__, allow_nan=False,skipkeys=True))

    # 前序遍历
    preOrder(tree)
