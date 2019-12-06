import tensorflow as tf
from tensorflow.keras import backend as K
import numpy as np

name = 'metrics'

# R^2
# https://www.kaggle.com/c/mercedes-benz-greener-manufacturing/discussion/34019
def R_squared(y_true, y_pred):
    
    numpy_ = True if(type(y_true).__name__ == 'ndarray') else False

    if(numpy_):
        y_true_mean = y_true.mean()
        y_pred_mean = y_pred.mean()

        sum_num = np.sum((y_pred-y_pred_mean)*y_true)
        numerator = np.square(sum_num)

        denominator = np.sum(np.square(y_pred - y_pred_mean)) \
            * np.sum(np.square(y_true - y_true_mean))
    
    else:
        y_true_mean = K.mean(y_true)
        y_pred_mean = K.mean(y_pred)

        numerator = K.square(K.sum((y_pred-y_pred_mean)*y_true))
        denominator = K.sum(K.square(y_pred - y_pred_mean))*K.sum(K.square(y_true - y_true_mean))

    
    R2 = numerator/denominator
    
    return R2

# Factor of two
# https://link.springer.com/content/pdf/10.1007/s00703-003-0070-7.pdf
def fac2(y_true, y_pred, to_numpy=False):
    min_ = 0.5
    max_ = 2

    division = tf.math.divide_no_nan(y_pred, y_true)

    greater_min = tf.greater_equal(division, min_)
    less_max = tf.less_equal(division, max_)

    # greater_min = tf.cast(greater_min, tf.float32)
    # less_max = tf.cast(less_max, tf.float32)

    res = tf.equal(greater_min, less_max)
    res = tf.cast(res, tf.float32)

    fac_2 = tf.reduce_mean(res)

    return K.get_value(fac_2) if to_numpy else fac_2

# Pearson r
# https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.pearsonr.html
# https://stackoverflow.com/questions/46619869/how-to-specify-the-correlation-coefficient-as-the-loss-function-in-keras
def pearson_r(y_true, y_pred):

    numpy_ = True if(type(y_true).__name__ == 'ndarray') else False

    if(numpy_):
        y_true_mean = y_true.mean()
        y_pred_mean = y_pred.mean()

        diff_yt = y_true - y_true_mean
        diff_yp = y_pred - y_pred_mean

        numerator = np.sum((diff_yt) * (diff_yp))
        denominator = np.sqrt(np.sum(np.square(diff_yt))) * np.sqrt(np.sum(np.square(diff_yp)))
    else:

        y_true_mean = K.mean(y_true)
        y_pred_mean = K.mean(y_pred)

        diff_yt = y_true - y_true_mean
        diff_yp = y_pred - y_pred_mean

        numerator = K.sum((diff_yt) * (diff_yp))
        denominator = K.sqrt(K.sum(K.square(diff_yt))) * K.sqrt(K.sum(K.square(diff_yp)))

    
    r = numerator/denominator

    return r
