from pgmpy.models import BayesianModel
from pgmpy.factors.discrete import TabularCPD

# 利用ByesianModel函数定义网络图
# 中括号内为整个图，小括号内前一项代表有向连接的起点，后一项代表终点

cancer_model = BayesianModel([('Pollution', 'Cancer'),
                              ('Smoker', 'Cancer'),
                              ('Cancer', 'Xray'),
                              ('Cancer', 'Dyspnoea')])

# 利用TabularCPD函数赋予结点具体的概率参数
# cariable：参数名
# variable_card：该结点可能出现的结果数目
# values：概率值
cpd_poll = TabularCPD(variable='Pollution',
                      variable_card=2,
                      values=[[0.9], [0.1]])
cpd_smoke = TabularCPD(variable='Smoker',
                       variable_card=2,
                       values=[[0.3], [0.7]])

# evidence:可以理解为条件概率中的条件
cpd_cancer = TabularCPD(variable='Cancer',
                        variable_card=2,
                        values=[[0.03, 0.05, 0.001, 0.02],
                                [0.97, 0.95, 0.999, 0.98]],
                        evidence=['Smoker', 'Pollution'],
                        evidence_card=[2, 2])
cpd_xray = TabularCPD(variable='Xray',
                      variable_card=2,
                      values=[[0.9, 0.2], [0.1, 0.8]],
                      evidence=['Cancer'],
                      evidence_card=[2])
cpd_dysp = TabularCPD(variable='Dyspnoea',
                      variable_card=2,
                      values=[[0.65, 0.3], [0.35, 0.7]],
                      evidence=['Cancer'],
                      evidence_card=[2])

# 利用add_cpds函数将参数与图连接起来
cancer_model.add_cpds(cpd_poll, cpd_smoke, cpd_cancer, cpd_xray, cpd_dysp)

# 检查模型是否合理，True代表合理
print(cancer_model.check_model())

# is_active_trail函数检验两个结点之间是否有有向连接
print(cancer_model.is_active_trail('Pollution', 'Smoker'))

# 在is_active_trail函数中，设置observed参数，表示两个结点能否通过observed结点实现连接
print(cancer_model.is_active_trail('Pollution', 'Smoker', observed=['Cancer']))