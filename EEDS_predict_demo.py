import cv2
import EEDS
import numpy

IMG_NAME = "./butterfly_GT.bmp"
INPUT_NAME = "input.jpg"
OUTPUT_NAME = "EEDS_pre.jpg"

scale = 2

label = cv2.imread(IMG_NAME)
shape = label.shape

img = cv2.resize(label, (shape[1] / scale, shape[0] / scale), cv2.INTER_CUBIC)
cv2.imwrite(INPUT_NAME, img)

model = EEDS.model_EEDS()
model.load_weights("EEDS_check.h5")

img = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
Y = numpy.zeros((1, img.shape[0], img.shape[1], 1))
Y[0, :, :, 0] = img[:, :, 0].astype(float) / 255.
img = cv2.cvtColor(label, cv2.COLOR_BGR2YCrCb)

pre = model.predict(Y, batch_size=1) * 255.
pre[pre[:] > 255] = 255
pre = numpy.uint8(pre)
img[:, :, 0] = pre[0, :, :, 0]
img = cv2.cvtColor(img, cv2.COLOR_YCrCb2BGR)
cv2.imwrite(OUTPUT_NAME, img)

# psnr calculation:
im1 = cv2.imread(IMG_NAME, cv2.IMREAD_COLOR)
im1 = cv2.cvtColor(im1, cv2.COLOR_BGR2YCrCb)
im2 = cv2.imread(INPUT_NAME, cv2.IMREAD_COLOR)
im2 = cv2.cvtColor(im2, cv2.COLOR_BGR2YCrCb)
im2 = cv2.resize(im2, (img.shape[1], img.shape[0]))
cv2.imwrite("Bicubic.jpg", cv2.cvtColor(im2, cv2.COLOR_YCrCb2BGR))
im3 = cv2.imread(OUTPUT_NAME, cv2.IMREAD_COLOR)
im3 = cv2.cvtColor(im3, cv2.COLOR_BGR2YCrCb)

print "Bicubic:"
print cv2.PSNR(im1[:, :, 0], im2[:, :, 0])
print "EEDS:"
print cv2.PSNR(im1[:, :, 0], im3[:, :, 0])