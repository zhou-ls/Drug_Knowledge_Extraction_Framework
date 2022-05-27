# 数据相关的配置
# event_type = "example_dis"

train_file_path = 'data/drug-disease/train.txt'
test_file_path = 'data/drug-disease/test.txt'
# dev_file_path = "./data/%s.dev" % event_type

# BERT base
# config_path = 'bert/chinese_L-12_H-768_A-12/bert_config.json'
# checkpoint_path = 'bert/chinese_L-12_H-768_A-12/bert_model.ckpt'
# dict_path = 'bert/chinese_L-12_H-768_A-12/vocab.txt'
config_path = 'roberta/chinese_roberta_zh_l12/bert_config.json'
checkpoint_path = 'roberta/chinese_roberta_zh_l12/bert_model.ckpt'
dict_path = 'roberta/chinese_roberta_zh_l12/vocab.txt'

# 标注的原始excel实体关系表
re_excel = r'data\实体关系表.xlsx'

# 测试模型性能的数据集
evaluate_file = r'data\origin_data\test_已标注.txt'

# 实体关系类型
json_label2id = './data/drug-disease/label.json'

# 模型相关的配置
num_classes = 2
batch_size = 16
epoch = 50
MAX_SEQ_LEN = 128  # 输入的文本最大长度

# 模型名称，bert, albert，ernie，ELECTRA, bert_wwm, bert_wwm_ext, roberta ,roberta_wwm_large ...ad_train
model_name = 'roberta'
# start_at = True  # 是否开启对抗训练

# 画图路径
loss_acc_png = r"./png/loss_acc_png/%s_loss_acc.png" % model_name  # loss_accuracy图
p_r_png_save_path = fr'./png/pr_png/%s-p-r.png' % model_name  # 模型性能评价 p-r 图
bar_chart = './png/bar_chart/bar_chart.png'  # 数据集中各种实体关系数据数量直方图

# 模型保存路径
model_dir = f'./models/%s' % model_name  # 模型路径
filepath = "%s/ex-rel-{epoch:02d}-{val_accuracy:.4f}.h5" % model_dir  # 存放训练结果
log_path = r'./logs/%s/train.log' % model_name
