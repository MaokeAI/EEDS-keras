from keras.models import Model
from keras.layers import Conv2D, Input, Deconvolution2D, merge
from keras.optimizers import SGD, adam
import prepare_data as pd
import numpy
import math


def psnr(target, ref):
    # assume RGB image
    target_data = numpy.array(target, dtype=float)
    ref_data = numpy.array(ref, dtype=float)

    diff = ref_data - target_data
    diff = diff.flatten('C')

    rmse = math.sqrt(numpy.mean(diff ** 2.))

    return 20 * math.log10(255. / rmse)


def model_EES(input_col, input_row):
    _input = Input(shape=(input_col, input_row, 1), name='input')

    EES = Conv2D(nb_filter=8, nb_row=3, nb_col=3, init='he_normal',
                 activation='relu', border_mode='same', bias=True)(_input)
    EES = Deconvolution2D(nb_filter=16, nb_row=14, nb_col=14, output_shape=(None, input_col * 2, input_row * 2, 16),
                          subsample=(2, 2), border_mode='same', init='glorot_uniform', activation='relu')(EES)
    out = Conv2D(nb_filter=1, nb_row=5, nb_col=5, init='glorot_uniform', activation='relu', border_mode='same')(EES)

    model = Model(input=_input, output=out)
    # sgd = SGD(lr=0.0001, decay=0.005, momentum=0.9, nesterov=True)
    Adam = adam(lr=0.001)
    model.compile(optimizer=Adam, loss='mean_squared_error', metrics=['mean_squared_error'])
    return model


def model_EED(input_col, input_row):
    _input = Input(shape=(input_col, input_row, 1), name='input')

    Feature = Conv2D(nb_filter=64, nb_row=3, nb_col=3, init='glorot_uniform',
                     activation='relu', border_mode='same', bias=True)(_input)
    Feature = Conv2D(nb_filter=64, nb_row=3, nb_col=3, init='glorot_uniform',
                     activation='relu', border_mode='same', bias=True)(Feature)
    Feature3 = Conv2D(nb_filter=64, nb_row=3, nb_col=3, init='glorot_uniform',
                      activation='relu', border_mode='same', bias=True)(Feature)
    Feature_out = merge(inputs=[Feature, Feature3], mode='sum')

    # Upsampling
    Upsampling1 = Conv2D(nb_filter=8, nb_row=1, nb_col=1, init='glorot_uniform',
                         activation='relu', border_mode='same', bias=True)(Feature_out)
    Upsampling2 = Deconvolution2D(nb_filter=8, nb_row=14, nb_col=14,
                                  output_shape=(None, input_col * 2, input_row * 2, 8),
                                  subsample=(2, 2), border_mode='same',
                                  init='glorot_uniform', activation='relu')(Upsampling1)
    Upsampling3 = Conv2D(nb_filter=64, nb_row=1, nb_col=1, init='glorot_uniform',
                         activation='relu', border_mode='same', bias=True)(Upsampling2)

    # Mulyi-scale Reconstruction
    Reslayer1 = Conv2D(nb_filter=64, nb_row=3, nb_col=3, init='glorot_uniform',
                       activation='relu', border_mode='same', bias=True)(Upsampling3)
    Reslayer2 = Conv2D(nb_filter=64, nb_row=3, nb_col=3, init='glorot_uniform',
                       activation='relu', border_mode='same', bias=True)(Reslayer1)
    Block1 = merge(inputs=[Reslayer1, Reslayer2], mode='sum')

    Reslayer3 = Conv2D(nb_filter=64, nb_row=3, nb_col=3, init='glorot_uniform',
                       activation='relu', border_mode='same', bias=True)(Block1)
    Reslayer4 = Conv2D(nb_filter=64, nb_row=3, nb_col=3, init='glorot_uniform',
                       activation='relu', border_mode='same', bias=True)(Reslayer3)
    Block2 = merge(inputs=[Reslayer3, Reslayer4], mode='sum')

    # ***************//
    Multi_scale1 = Conv2D(nb_filter=16, nb_row=1, nb_col=1, init='glorot_uniform',
                          activation='relu', border_mode='same', bias=True)(Block2)
    Multi_scale2a = Conv2D(nb_filter=16, nb_row=1, nb_col=1, init='glorot_uniform',
                           activation='relu', border_mode='same', bias=True)(Multi_scale1)
    Multi_scale2b = Conv2D(nb_filter=16, nb_row=3, nb_col=3, init='glorot_uniform',
                           activation='relu', border_mode='same', bias=True)(Multi_scale1)
    Multi_scale2c = Conv2D(nb_filter=16, nb_row=5, nb_col=5, init='glorot_uniform',
                           activation='relu', border_mode='same', bias=True)(Multi_scale1)
    Multi_scale2d = Conv2D(nb_filter=16, nb_row=7, nb_col=7, init='glorot_uniform',
                           activation='relu', border_mode='same', bias=True)(Multi_scale1)
    Multi_scale2 = merge(inputs=[Multi_scale2a, Multi_scale2b, Multi_scale2c, Multi_scale2d], mode='concat')

    out = Conv2D(nb_filter=1, nb_row=1, nb_col=1, init='glorot_uniform',
                 activation='relu', border_mode='same', bias=True)(Multi_scale2)
    model = Model(input=_input, output=out)

    Adam = adam(lr=0.001)
    model.compile(optimizer=Adam, loss='mean_squared_error', metrics=['mean_squared_error'])

    return model


def model_EEDS(input_col, input_row):
    _input = Input(shape=(input_col, input_row, 1), name='input')
    EES = model_EES(input_col, input_row)(_input)
    EED = model_EED(input_col, input_row)(_input)
    _EEDS = merge(inputs=[EED, EES], mode='sum')

    model = Model(input=_input, output=_EEDS)
    Adam = adam(lr=0.001)
    model.compile(optimizer=Adam, loss='mean_squared_error', metrics=['mean_squared_error'])
    return model


def EEDS_train():
    EEDS = model_EEDS(input_col=48, input_row=48)
    data, label = pd.read_training_data("./little_train.h5")
    EEDS.fit(data, label, batch_size=256, nb_epoch=100)
    EEDS.save_weights("EEDS8_model_adam100.h5")


def EES_train():
    EES = model_EES(input_col=48, input_row=48)
    data, label = pd.read_training_data("./train.h5")
    EES.fit(data, label, batch_size=256, nb_epoch=200)
    EES.save_weights("EES_model_adam200.h5")


def EEDS_predict():
    EEDS = model_EEDS(input_col=128, input_row=128)
    EEDS.load_weights("EEDS8_model_adam100.h5")
    IMG_NAME = "butterfly_GT.bmp"
    INPUT_NAME = "input.jpg"
    OUTPUT_NAME = "EEDS8_adam100.jpg"

    import cv2
    img = cv2.imread(IMG_NAME)
    # img = img[:96, :96, :]
    shape = img.shape
    img = cv2.resize(img, (shape[1] / 2, shape[0] / 2), cv2.INTER_CUBIC)
    cv2.imwrite(INPUT_NAME, img)

    img = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    Y = numpy.zeros((1, img.shape[0], img.shape[1], 1))
    Y[0, :, :, 0] = img[:, :, 0]
    img = cv2.resize(img, (shape[1], shape[0]), cv2.INTER_CUBIC)

    pre = EEDS.predict(Y, batch_size=1)
    pre[pre[:] > 255] = 255
    pre[pre[:] < 0] = 0
    pre = pre.astype(numpy.uint8)
    img[:, :, 0] = pre[0, :, :, 0]
    img = cv2.cvtColor(img, cv2.COLOR_YCrCb2BGR)
    cv2.imwrite(OUTPUT_NAME, img)

    # psnr calculation:
    im1 = cv2.imread(IMG_NAME, cv2.IMREAD_COLOR)
    im1 = cv2.cvtColor(im1, cv2.COLOR_BGR2YCrCb)
    im2 = cv2.imread(INPUT_NAME, cv2.IMREAD_COLOR)
    im2 = cv2.cvtColor(im2, cv2.COLOR_BGR2YCrCb)
    im2 = cv2.resize(im2, (img.shape[1], img.shape[0]))
    im3 = cv2.imread(OUTPUT_NAME, cv2.IMREAD_COLOR)
    im3 = cv2.cvtColor(im3, cv2.COLOR_BGR2YCrCb)

    print "Bicubic:"
    print cv2.PSNR(im1, im2)  # [:, :, 0]
    print "EEDS:"
    print cv2.PSNR(im1, im3)


def EES_predict():
    EES = model_EES(input_col=128, input_row=128)
    EES.load_weights("EES_model_adam200.h5")
    IMG_NAME = "butterfly_GT.bmp"
    INPUT_NAME = "input.jpg"
    OUTPUT_NAME = "EES_pre_adam200.jpg"

    import cv2
    img = cv2.imread(IMG_NAME)
    # img = img[:96, :96, :]
    shape = img.shape
    img = cv2.resize(img, (shape[1] / 2, shape[0] / 2), cv2.INTER_CUBIC)
    cv2.imwrite(INPUT_NAME, img)

    img = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    Y = numpy.zeros((1, img.shape[0], img.shape[1], 1))
    Y[0, :, :, 0] = img[:, :, 0]
    img = cv2.resize(img, (shape[1], shape[0]), cv2.INTER_CUBIC)

    pre = EES.predict(Y, batch_size=1)
    pre[pre[:] > 255] = 255
    pre[pre[:] < 0] = 0
    pre = pre.astype(numpy.uint8)
    img[:, :, 0] = pre[0, :, :, 0]
    img = cv2.cvtColor(img, cv2.COLOR_YCrCb2BGR)
    cv2.imwrite(OUTPUT_NAME, img)

    # psnr calculation:
    im1 = cv2.imread(IMG_NAME, cv2.IMREAD_COLOR)
    # im1 = cv2.cvtColor(im1, cv2.COLOR_BGR2YCrCb)
    im2 = cv2.imread(INPUT_NAME, cv2.IMREAD_COLOR)
    # im2 = cv2.cvtColor(im2, cv2.COLOR_BGR2YCrCb)
    im2 = cv2.resize(im2, (img.shape[1], img.shape[0]))
    im3 = cv2.imread(OUTPUT_NAME, cv2.IMREAD_COLOR)
    # im3 = cv2.cvtColor(im3, cv2.COLOR_BGR2YCrCb)

    print "Bicubic:"
    print cv2.PSNR(im1, im2)
    print "EES:"
    print cv2.PSNR(im1, im3)


if __name__ == "__main__":
    # EES_train()
    # EES_predict()
    EEDS_train()
    EEDS_predict()
