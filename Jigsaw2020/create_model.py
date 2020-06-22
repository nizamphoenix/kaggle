

from tensorflow.keras.metrics import AUC
from tensorflow.keras.layers import Dense, Input, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint
def build_model(transformer, max_len):
    """
    https://www.kaggle.com/xhlulu/jigsaw-tpu-distilbert-with-huggingface-and-keras
    """
    input_word_ids = Input(shape=(max_len,), dtype=tf.int32, name="input_word_ids")
    sequence_output = transformer(input_word_ids)[0]
    cls_token = sequence_output[:, 0, :]#cls_token is a vector of length 768, marginalised against other 2 dimensions
    x = Dense(256, activation=gelu)(cls_token)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    x = Dense(32, activation=gelu)(cls_token)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
    out = Dense(1, activation='sigmoid')(x)
    model = Model(inputs=input_word_ids, outputs=out)
    model.compile(Adam(lr=1e-5), loss='binary_crossentropy', metrics=['accuracy',AUC(name='auc')])
    return model


%%time
from transformers import TFAutoModel
from transformers import TFAutoModelForSequenceClassification
with strategy.scope():
    transformer_layer = TFAutoModel.from_pretrained('jplu/tf-xlm-roberta-base')
    transformer_layer = TFAutoModelForSequenceClassification.from_pretrained('jplu/tf-xlm-roberta-base')
    model = build_model(transformer_layer, max_len=MAX_LEN)
model.summary()



# def build_model_working(transformer, max_len):
#     """
#     from transformers import TFAutoModelForSequenceClassification
#     transformer = TFAutoModelForSequenceClassification.from_pretrained('jplu/tf-xlm-roberta-base')
#     """
#     input_word_ids = Input(shape=(max_len,), dtype=tf.int32, name="input_word_ids")
#     sequence_output = transformer(input_word_ids)[0]
#     out = Dense(1, activation='sigmoid')(sequence_output)
#     model = Model(inputs=input_word_ids, outputs=out)
#     model.compile(Adam(lr=1e-3), loss='binary_crossentropy', metrics=['accuracy',AUC(name='auc')])
#     return model
