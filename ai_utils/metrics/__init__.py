import tensorflow as tf
from tensorflow.keras import backend as K

name = 'metrics'

# R^2
# https://www.kaggle.com/c/mercedes-benz-greener-manufacturing/discussion/34019
def R_squared(y_true, y_pred, to_numpy=False):
    SS_res =  K.sum(K.square(y_true - y_pred))
    SS_tot = K.sum(K.square(y_true - K.mean(tf.constant(y_true))))

    R2 = (1 - SS_res/(SS_tot + K.epsilon()))
    return K.get_value(R2) if to_numpy else R2

# Factor of two
# https://link.springer.com/content/pdf/10.1007/s00703-003-0070-7.pdf
def fac2(y_true, y_pred, to_numpy=False):
    min_ = 0.5
    max_ = 2

    division = tf.divide(y_pred, y_true)

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
def pearson_r(y_true, y_pred, to_numpy=False):
    x = y_true
    y = y_pred
    mx = K.mean(x)
    my = K.mean(y)
    xm, ym = x-mx, y-my
    r_num = K.sum(tf.multiply(xm,ym))
    r_den = K.sqrt(tf.multiply(K.sum(K.square(xm)), K.sum(K.square(ym))))
    r = r_num / r_den

    return K.get_value(r) if to_numpy else r
