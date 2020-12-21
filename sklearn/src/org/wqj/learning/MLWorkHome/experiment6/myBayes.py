from pgmpy.readwrite import BIFReader
from pgmpy.inference import VariableElimination
import os

curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = curPath[:curPath.find("sklearn\\") + len("sklearn\\")]
dataPath = rootPath + "Input/MLWorkHome/experiment6/img/asia.bif"
reader = BIFReader(dataPath)
asia_model = reader.get_model()
# 通过nodes函数可以查看模型中有哪些结点
print(asia_model.nodes())
# NodeView(('xray', 'bronc', 'asia', 'dysp', 'lung', 'either', 'smoke', 'tub'))
# 练习1   在下面的单元格中，实现判断，判断tub结点和either结点之间是否存在有向连接:
print("练习1:")
print(asia_model.is_active_trail('tub', 'either'))

# 练习2   在下面的单元格中，实现判断，判断tub结点和dysp结点之间能否通过either结点有向连接:
print("练习2:")
print(asia_model.is_active_trail('tub', 'dysp', observed=['either']))

asia_infer = VariableElimination(asia_model)
# 给出当smoke为0时，bronc的概率分布情况
q = asia_infer.query(variables=['bronc'], evidence={'smoke': 0})
print(q['bronc'])

# 练习3   在下面的单元格中，实现查询，当either为1时，xray的概率分布情况:
print("练习3:")
asia_infer2 = VariableElimination(asia_model)
p = asia_infer2.query(variables=['xray'], evidence={'either': 1})
print(p['xray'])
