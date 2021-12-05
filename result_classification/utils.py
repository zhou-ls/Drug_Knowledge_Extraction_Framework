# 数据相关的配置
train_file_path = 'data/train.txt'
test_file_path = 'data/test.txt'

# 标注的原始excel实体关系表
re_excel = r'data\实体关系表.xlsx'

# 测试模型性能的数据集
evaluate_file = r'data\origin_data\test_labeled.txt'

# 实体关系类型
json_label2id = './data/rel_dict.json'

# 模型相关的配置
batch_size = 16
epoch = 100
MAX_SEQ_LEN = 80  # 输入的文本最大长度

# 模型名称
model_name = 'bert_wwm'
num_classes = 5
# 画图路径
loss_acc_png = r"./png/loss_acc_png/%s_loss_acc.png" % model_name  # loss_accuracy图
p_r_png_save_path = fr'./png/pr_png/%s-p-r.png' % model_name  # 模型性能评价 p-r 图
bar_chart = './png/bar_chart/bar_chart.png'  # 数据集中各种实体关系数据数量直方图

model_dir = f'./models/%s' % model_name  # 模型路径
filepath = "%s/ex-rel-{epoch:02d}-{val_accuracy:.4f}.h5" % model_dir  # 存放训练结果

log_path = r'./logs/%s/train.log' % model_name
