# 药品知识抽取框架

目前放置 **EPC** 及 **药品相互作用结果的细粒度关系抽取**的代码

文件目录如下：

```shell
├─EPC
│  │  att.py  # 注意力机制
│  │  data_into_train_test.py # 得到训练数据
│  │  draw_picture.py  # 训练过程中画图
│  │  load_data.py  # 加载数据
│  │  model_predict.py  # 模型预测
│  │  model_train.py  # 模型训练
│  │  relation_bar_chart.py # 实体关系分析
│  │  requirements.txt  # 项目依赖模块
│  │  tmp_graph11  # 句向量生成，初次使用加载graph,并生成该文件，再次使用时加快速度
│  │  utils.py  # 模型配置
│  │  
│  ├─bert_wwm  # bert_wwm配置
│  │  │  args.py
│  │  │  extract_feature.py
│  │  │  graph.py
│  │  │  modeling.py
│  │  │  optimization.py
│  │  │  tokenization.py
│  │  │  __init__.py
│  │  │  
│  │  ├─bert_wwm
│  │         bert_config.json
│  │         bert_model.ckpt.data-00000-of-00001
│  │         bert_model.ckpt.index
│  │         bert_model.ckpt.meta
│  │         vocab.txt             
│  │          
│  ├─data
│  │  │  rel_dict.json # 关系类型
│  │  │  test.txt # 测试集
│  │  │  train.txt # 训练集
│  │  │  实体关系表.xlsx
│  │  │  
│  │  └─origin_data  # 原始人工标注数据
│  │          test_labeled.txt
│  │          train_labeled.txt
│  │          write_excel.py
│  │          
│  ├─logs # 训练输出log
│  │  └─bert_wwm
│  │          train.log
│  │          
│  ├─models # 保存的模型文件
│  │  └─bert_wwm
│  │          ex-rel-12-0.9880.h5
│  │          
│  └─png # 保存图片
│      ├─bar_chart
│      │      bar_chart.png
│      │      
│      ├─loss_acc_png
│      │      bert_wwm_loss_acc.png
│      │      
│      └─pr_png
│              bert_wwm-p-r.png
│              
└─result_classification
    │  att.py # 注意力机制
    │  data_into_train_test.py # 得到训练数据
    │  draw_picture.py # 训练过程中画图
    │  load_data.py # 加载数据
    │  model_predict.py  # 模型预测
    │  model_train.py # 模型训练
    │  relation_bar_chart.py # 实体关系分析
    │  requirements.txt # 项目依赖模块
    │  tmp_graph11 # 句向量生成，初次使用加载graph,并生成该文件，再次使用时加快速度
    │  utils.py # 模型配置
    │  
    ├─bert_wwm # bert_wwm配置
    │  │  args.py
    │  │  extract_feature.py
    │  │  graph.py
    │  │  modeling.py
    │  │  optimization.py
    │  │  tokenization.py
    │  │  __init__.py
    │  │  
    │  ├─bert_wwm
    │        bert_config.json
    │        bert_model.ckpt.data-00000-of-00001
    │        bert_model.ckpt.index
    │        bert_model.ckpt.meta
    │        vocab.txt
    │        
    │          
    ├─data
    │  │  rel_dict.json # 关系类型
    │  │  test.txt # 测试集
    │  │  train.txt # 训练集
    │  │  实体关系表.xlsx
    │  │  
    │  └─origin_data # 原始标注数据
    │          test_labeled.txt
    │          train_labeled.txt
    │          write_excel.py
    │          
    ├─logs # 训练输出log
    │  └─bert_wwm
    │          train.log
    │          
    ├─models # 保存的模型文件
    │  └─bert_wwm
    │          ex-rel-03-0.8465.h5
    │          
    └─png # 保存图片
        ├─bar_chart
        │      bar_chart.png
        │      
        ├─loss_acc_png
        │      bert_wwm_loss_acc.png
        │      
        └─pr_png
                bert_wwm-p-r.png

```

代码基于BERT-WWM中文预训练模型，需要下载BERT-WWM的中文预训练模型放入bert_wwm目录中，参考：https://github.com/tiffen/Chinese-BERT-wwm

相关文件说明如上面文件中注释部分所示，按照 `\data`目录下训练集的格式获取相关数据集，并在`utils.py`执行中按照实际情况修改相关参数，然后执行

```shell
> cd EPC 
> python model_train.py
```

即可训练模型，

```python
> python model_predict.py
```

即可进行模型预测输出，即输出实体的角色或实体对的关系，待预测输入的数据格式参考`.\EPC\data\origin_data\test_labeled.txt`目录下的文件格式，也可根据实际情况需要稍加修改`model_predict.py`，预测不同格式的文本

