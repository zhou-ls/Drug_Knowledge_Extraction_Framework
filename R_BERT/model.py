# -*- coding: utf-8 -*-

from tensorflow.keras.models import Model
from tensorflow.keras.utils import plot_model
from tensorflow.keras.layers import Input, Lambda, Dense, Dropout, concatenate, Dot
from tensorflow.keras.layers import GlobalMaxPool1D, Multiply, Add, Activation
from keras_bert import load_trained_model_from_checkpoint

# model structure of R-BERT
class RBERT(object):
    def __init__(self, config_path, checkpoint_path, maxlen, num_labels):
        self.config_path = config_path
        self.checkpoint_path = checkpoint_path
        self.maxlen = maxlen
        self.num_labels = num_labels

    def create_model(self):
        # BERT model
        bert_model = load_trained_model_from_checkpoint(self.config_path, self.checkpoint_path, seq_len=None)
        for layer in bert_model.layers:
            layer.trainable = True
        x1_in = Input(shape=(self.maxlen,))
        x2_in = Input(shape=(self.maxlen,))
        bert_layer = bert_model([x1_in, x2_in])

        # get three vectors
        cls_layer = Lambda(lambda x: x[:, 0])(bert_layer)    # 取出[CLS]对应的向量
        e1_mask = Input(shape=(self.maxlen,))
        e2_mask = Input(shape=(self.maxlen,))
        e1_layer = self.entity_average(bert_layer, e1_mask)  # 取出实体1对应的向量
        e2_layer = self.entity_average(bert_layer, e2_mask)  # 取出实体2对应的向量

        # dropout -> linear -> concatenate
        output_dim = cls_layer.shape[-1].value
        cls_fc_layer = self.crate_fc_layer(cls_layer, output_dim, dropout_rate=0.1)
        e1_fc_layer = self.crate_fc_layer(e1_layer, output_dim, dropout_rate=0.1)
        e2_fc_layer = self.crate_fc_layer(e2_layer, output_dim, dropout_rate=0.1)
        # 将concatenate改成gated mechanism
        # concat_layer = concatenate([cls_fc_layer, e1_fc_layer, e2_fc_layer], axis=-1)
        cls_e1_fusion = Add()([cls_fc_layer, e1_fc_layer])
        cls_e2_fusion = Add()([cls_fc_layer, e2_fc_layer])
        concat_layer = self.gated_mechanism(cls_e1_fusion, cls_e2_fusion, output_dim)

        # FC layer for classification
        # fc_layer = Dense(200, activation="relu")(concat_layer)
        output = Dense(self.num_labels, activation="softmax")(concat_layer)
        model = Model([x1_in, x2_in, e1_mask, e2_mask], output)
        model.summary()
        return model

    @staticmethod
    def crate_fc_layer(input_layer, output_dim, dropout_rate=0.0, activation_func="tanh"):
        dropout_layer = Dropout(rate=dropout_rate)(input_layer)
        linear_layer = Dense(output_dim, activation=activation_func)(dropout_layer)
        return linear_layer

    @staticmethod
    def entity_average(hidden_output, e_mask):
        """
        Average the entity hidden state vectors (H_i ~ H_j)
        :param hidden_output: BERT hidden output
        :param e_mask:
                e.g. e_mask[0] == [0, 0, 0, 1, 1, 1, 0, 0, ... 0]/num_of_ones
        :return: entity average layer
        """
        avg_layer = Dot(axes=1)([e_mask, hidden_output])
        return avg_layer

    @staticmethod
    def gated_mechanism(x1, x2, HIDDEN_STATE_NUMBER):
        vector1 = Dense(HIDDEN_STATE_NUMBER, input_dim=HIDDEN_STATE_NUMBER)(x1)
        vector2 = Dense(HIDDEN_STATE_NUMBER, input_dim=HIDDEN_STATE_NUMBER)(x2)
        sigmoid_value = Activation(activation="sigmoid")(Add()([vector1, vector2]))
        tmp1 = Multiply()([sigmoid_value, x1])
        tmp2 = Multiply()([Lambda(lambda x: 1 - x)(sigmoid_value), x2])
        fusion_vector = Add()([tmp1, tmp2])
        return fusion_vector


if __name__ == '__main__':
    model_config = "./chinese_L-12_H-768_A-12/bert_config.json"
    model_checkpoint = "./chinese_L-12_H-768_A-12/bert_model.ckpt"
    model = RBERT(model_config, model_checkpoint, 128, 14).create_model()
    plot_model(model, to_file="model_structure.png")