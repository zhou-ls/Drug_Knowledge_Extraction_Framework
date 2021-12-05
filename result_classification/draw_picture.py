# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
from sklearn.metrics import average_precision_score, precision_recall_curve
from funcsigs import signature

from utils import model_name, p_r_png_save_path, loss_acc_png
"""
绘制各种图
"""


def draw_loss_acc(history):
    # 绘制loss和acc图像
    plt.subplot(2, 1, 1)
    epochs = len(history.history['loss'])
    plt.plot(range(epochs), history.history['loss'], label='loss')
    plt.plot(range(epochs), history.history['val_loss'], label='val_loss')
    plt.legend()

    plt.subplot(2, 1, 2)
    epochs = len(history.history['accuracy'])
    plt.plot(range(epochs), history.history['accuracy'], label='acc')
    plt.plot(range(epochs), history.history['val_accuracy'], label='val_acc')
    plt.legend()
    plt.savefig(loss_acc_png)


def draw_p_r(y_true, y_score):
    average_precision = average_precision_score(y_true, y_score)
    print('PR curve area:' + str(average_precision))
    precision, recall, _ = precision_recall_curve(y_true, y_score)
    # In matplotlib < 1.5, plt.fill_between does not have a 'step' argument
    step_kwargs = ({'step': 'post'}
                   if 'step' in signature(plt.fill_between).parameters
                   else {})

    plt.step(recall, precision, color='coral', alpha=0.2, where='post')
    # plt.fill_between(recall, precision, alpha=0.2, color='b', **step_kwargs)

    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.0])
    plt.xlim([0.2, 1.0])
    plt.title(model_name + ' Precision-Recall curve: AP={0:0.4f}'.format(average_precision))
    plt.figure(1)  # plt.figure(1)是新建一个名叫 Figure1的画图窗口
    plt.plot(recall, precision)
    plt.savefig(p_r_png_save_path)
    plt.show()

