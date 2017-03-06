from keras.models import Model
from keras.layers import Conv2D, Input, Deconvolution2D, merge
from keras.optimizers import SGD, adam
import prepare_data as pd
import numpy
import cv2


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


def EES_train():
    EES = model_EES(input_col=48, input_row=48)
    data, label = pd.read_training_data("./little_train.h5")
    EES.fit(data, label, batch_size=256, nb_epoch=100)
    EES.save_weights("EES_model_adam100.h5")


def EES_predict():
    EES = model_EES(input_col=128, input_row=128)
    EES.load_weights("EES_model_adam100.h5")
    IMG_NAME = "butterfly_GT.bmp"
    INPUT_NAME = "input.jpg"
    OUTPUT_NAME = "EES_pre_adam100.jpg"

    img = cv2.imread(IMG_NAME)
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
    im1 = cv2.cvtColor(im1, cv2.COLOR_BGR2YCrCb)
    im2 = cv2.imread(INPUT_NAME, cv2.IMREAD_COLOR)
    im2 = cv2.cvtColor(im2, cv2.COLOR_BGR2YCrCb)
    im2 = cv2.resize(im2, (img.shape[1], img.shape[0]))
    im3 = cv2.imread(OUTPUT_NAME, cv2.IMREAD_COLOR)
    im3 = cv2.cvtColor(im3, cv2.COLOR_BGR2YCrCb)

    print "Bicubic:"
    print cv2.PSNR(im1[:, :, 0], im2[:, :, 0])
    print "EES:"
    print cv2.PSNR(im1[:, :, 0], im3[:, :, 0])


##*******************************************************************************************************************//
from math import sqrt

import matplotlib.pyplot as plt
from keras import backend as K


# Function by gcalmettes from http://stackoverflow.com/questions/11159436/multiple-figures-in-a-single-window
def plot_figures(figures, nrows=1, ncols=1, titles=False):
    """Plot a dictionary of figures.
    Parameters
    ----------
    figures : <title, figure> dictionary
    ncols : number of columns of subplots wanted in the display
    nrows : number of rows of subplots wanted in the figure
    """
    fig, axeslist = plt.subplots(ncols=ncols, nrows=nrows)
    for ind, title in enumerate(sorted(figures.keys(), key=lambda s: int(s[3:]))):
        axeslist.ravel()[ind].imshow(figures[title], cmap=plt.gray())
        if titles:
            axeslist.ravel()[ind].set_title(title)

    for ind in range(nrows*ncols):
        axeslist.ravel()[ind].set_axis_off()

    if titles:
        plt.tight_layout()
    plt.show()


def get_dim(num):
    """
    Simple function to get the dimensions of a square-ish shape for plotting
    num images
    """
    s = sqrt(num)
    if round(s) < s:
        return (int(s), int(s)+1)
    else:
        return (int(s)+1, int(s)+1)


def feature_map_visilization(model, _input):
    # Get the convolutional layers
    conv_layers = [layer for layer in model.layers if isinstance(layer, Conv2D)]

    # Use a keras function to extract the conv layer data
    convout_func = K.function([model.layers[0].input, K.learning_phase()], [layer.output for layer in conv_layers])
    conv_imgs_filts = convout_func([_input, 0])
    # Also get the prediction so we know what we predicted
    predictions = model.predict(_input)

    imshow = plt.imshow  # alias

    # Show the original image
    plt.title("Image used:")
    imshow(_input[0, :, :, 0], cmap='gray')
    plt.tight_layout()
    plt.show()

    # Plot the filter images
    for i, conv_imgs_filt in enumerate(conv_imgs_filts):
        conv_img_filt = conv_imgs_filt[0]
        print("Visualizing Convolutions Layer %d" % i)
        # Get it ready for the plot_figures function
        fig_dict = {'flt{0}'.format(i): conv_img_filt[:, :, i] for i in range(conv_img_filt.shape[-1])}
        plot_figures(fig_dict, *get_dim(len(fig_dict)))

    cv2.waitKey(0)


def vilization_and_show():
    model = model_EES(input_col=128, input_row=128)
    model.load_weights("EES_model_adam100.h5")
    IMG_NAME = "butterfly_GT.bmp"
    INPUT_NAME = "input.jpg"
    # OUTPUT_NAME = "EES_pre_adam100.jpg"

    img = cv2.imread(IMG_NAME)
    shape = img.shape
    img = cv2.resize(img, (shape[1] / 2, shape[0] / 2), cv2.INTER_CUBIC)
    cv2.imwrite(INPUT_NAME, img)

    img = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    Y = numpy.zeros((1, img.shape[0], img.shape[1], 1))
    Y[0, :, :, 0] = img[:, :, 0]

    feature_map_visilization(model, Y)


if __name__ == "__main__":
    EES_train()
    EES_predict()
    vilization_and_show()
