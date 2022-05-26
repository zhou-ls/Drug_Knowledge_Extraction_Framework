import logging

import numpy as np
from keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import classification_report

from load_data import train_df, test_df
from keras.utils import to_categorical
from keras.models import Model
from keras.optimizers import Adam
from keras.layers import Input, BatchNormalization, Dense
import matplotlib.pyplot as plt

from roberta.extract_feature import BertVector
# from bert.extract_feature import BertVector

# 读取文件并进行转换
bert_model = BertVector(pooling_strategy="REDUCE_MEAN", max_seq_len=256)
print('begin encoding')
f = lambda text: bert_model.encode([text])["encodes"][0]
train_df['x'] = train_df['text'].apply(f)
test_df['x'] = test_df['text'].apply(f)
print('end encoding')

x_train = np.array([vec for vec in train_df['x']])
x_test = np.array([vec for vec in test_df['x']])
y_train = np.array([vec for vec in train_df['label']])
y_test = np.array([vec for vec in test_df['label']])
print('x_train: ', x_train.shape)

# Convert class vectors to binary class matrices.
num_classes = 2
y_train = to_categorical(y_train, num_classes)
y_test = to_categorical(y_test, num_classes)

# 创建模型
x_in = Input(shape=(768, ))
x_out = Dense(32, activation="relu")(x_in)
x_out = BatchNormalization()(x_out)
x_out = Dense(num_classes, activation="softmax")(x_out)
model = Model(inputs=x_in, outputs=x_out)
print(model.summary())

model.compile(loss='categorical_crossentropy',
              optimizer=Adam(),
              metrics=['accuracy'])

# 模型训练以及评估
early_stopping = EarlyStopping(monitor='val_accuracy', patience=5, mode='max')
checkpoint = ModelCheckpoint("model/roberta2/result-drug-drug2-event_type_classify.h5", monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')
history = model.fit(x_train, y_train, validation_data=(x_test, y_test), batch_size=16, epochs=50, callbacks=[checkpoint,early_stopping])
model.save('model/roberta2/result-drug-drug2-event_type_classify.h5')
# print(model.evaluate(x_test, y_test))
print('在测试集上的效果：', model.evaluate(x_test, y_test))
y_pred = model.predict(x_test, batch_size=16)
# print(y_test)  # 标签的one_hot编码，二维向量
# print(y_pred)  # 实际预测的置信度，二维向量
print(classification_report(y_true=y_test.argmax(axis=1), y_pred=y_pred.argmax(axis=1), digits=4))
logging.basicConfig(level=logging.DEBUG,  # 控制台打印的日志级别
                    filename='./logs/roberta/train.log',
                    filemode='a',  # 模式，有w和a，w就是写模式，每次都会重新写日志，覆盖之前的日志,a是追加模式，默认如果不写的话，就是追加模式
                    format='%(asctime)s - %(levelname)s: %(message)s'  # 日志格式
                    )
logging.info('\n' + classification_report(y_true=y_test.argmax(axis=1), y_pred=y_pred.argmax(axis=1), digits=4))
# 绘制loss和acc图像
plt.subplot(2, 1, 1)
epochs = len(history.history['loss'])
plt.plot(range(epochs), history.history['loss'], label='loss')
plt.plot(range(epochs), history.history['val_loss'], label='val_loss')
plt.legend()

plt.subplot(2, 1, 2)
epochs = len(history.history['accuracy'])
plt.plot(range(epochs), history.history['accuracy'], label='accuracy')
plt.plot(range(epochs), history.history['val_accuracy'], label='val_accuracy')
plt.legend()
plt.savefig("loss_acc.png")
