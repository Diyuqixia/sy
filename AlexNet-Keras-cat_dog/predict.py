import numpy as np
import utils
import cv2
import glob
from model.AlexNet import AlexNet
path = r'D:\python_project\AlexNet-Keras-master\data\image\test\dog_test\*.jpg'
wrong = 0
num = 0
if __name__ == "__main__":
    model = AlexNet()
    model.load_weights("./logs/train_fin.h5")
for i in glob.glob(path):
    img = cv2.imread(i)
    img_RGB = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    img_nor = img_RGB/255
    img_nor = np.expand_dims(img_nor,axis = 0)
    img_resize = utils.resize_image(img_nor,(224,224))
    num = num + 1
    result = np.argmax(model.predict(img_resize))
    print(utils.print_answer(result))

    if result== 0 :
      wrong = wrong + 1
acc = 1 - wrong/num
print("把狗错分为猫的个数：", wrong)
print("准确率为：",acc)