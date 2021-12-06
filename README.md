**[English](https://github.com/zhou-ls/Drug_Knowledge_Extraction_Framework)** | **[中文](https://github.com/zhou-ls/Drug_Knowledge_Extraction_Framework/blob/main/README_zh.md)**

# Drug knowledge extraction framework

Code available for entity pair calibration and fine-grained drug interaction extraction.
Folder structure as below:

```shell
├─EPC
│  │  att.py  # attention mechanism
│  │  data_into_train_test.py # get train data
│  │  draw_picture.py  # paint while training
│  │  load_data.py  # load dataset
│  │  model_predict.py  # make predictions
│  │  model_train.py  # train the model 
│  │  relation_bar_chart.py # entity relation extraction
│  │  requirements.txt  # dependants
│  │  tmp_graph11  # create sentence vectors, first usage takes longer time
│  │  utils.py  # model config
│  │  
│  ├─bert_wwm  # bertwwm config
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
│  │  │  rel_dict.json # relation types
│  │  │  test.txt # test dataset
│  │  │  train.txt # train dataset
│  │  │  实体关系表.xlsx #
│  │  │  
│  │  └─origin_data  # original manually labelled data
│  │          test_labeled.txt
│  │          train_labeled.txt
│  │          write_excel.py
│  │          
│  ├─logs # log output
│  │  └─bert_wwm
│  │          train.log
│  │          
│  ├─models # saved models
│  │  └─bert_wwm
│  │          ex-rel-12-0.9880.h5
│  │          
│  └─png # saved pics
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
    │  att.py # attention mechianism
    │  data_into_train_test.py # get training dataset
    │  draw_picture.py # draw while training
    │  load_data.py # load dataset
    │  model_predict.py  # make predictions
    │  model_train.py # model training
    │  relation_bar_chart.py # relation analysis
    │  requirements.txt # dependants
    │  tmp_graph11 # create sentence vectors, first usage takes longer time
    │  utils.py # model config
    │  
    ├─bert_wwm # bert_wwmconfig
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
    │  │  rel_dict.json # relation types
    │  │  test.txt # test dataset
    │  │  train.txt # train dataset
    │  │  实体关系表.xlsx
    │  │  
    │  └─origin_data # original manually labelled data
    │          test_labeled.txt
    │          train_labeled.txt
    │          write_excel.py
    │          
    ├─logs # log output
    │  └─bert_wwm
    │          train.log
    │          
    ├─models # saved models
    │  └─bert_wwm
    │          ex-rel-03-0.8465.h5
    │          
    └─png # saved pics
        ├─bar_chart
        │      bar_chart.png
        │      
        ├─loss_acc_png
        │      bert_wwm_loss_acc.png
        │      
        └─pr_png
                bert_wwm-p-r.png

```


code based on BERT-WWM for chinese, need to download the pre-training model into folder `bert_wwm`, see: https://github.com/tiffen/Chinese-BERT-wwm


Project file specification as above, use the format under `\data` folder for training dataset, change parameter configurations in the `utils.py` and execute below:

```shell
> cd EPC 
> python model_train.py
```


the above will train the model

```python
> python model_predict.py
```

the above will make predifctions, i.e., generate entity roles or entity pair relations, input format see `.\EPC\data\origin_data\test_labeled.txt`
you can also modify `model_predict.py` to make other types of predictions.