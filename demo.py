import os
import glob
import cv2
import pytesseract
from pytesseract import Output

my_path = "/media/cvpr/CM_1/COREMAX/testing/"

for img in glob.glob(my_path + '*.*'):
    d = pytesseract.image_to_data(img, output_type=Output.DICT, lang='kor')
    print(d.keys())
    n_boxes = len(d['text'])
    img_bgr_rgb = cv2.imread(img)
    file_Name = os.path.basename(img)
    image = img_bgr_rgb[:, :, ::-1]
    for i in range(n_boxes):
        if int(d['conf'][i]) > 60:
            (x, y, w, h) = (d['left'][i], d['top'][i], d['width'][i], d['height'][i])
            img = cv2.rectangle(img_bgr_rgb, (x, y), (x + w, y + h), (0, 255, 0), -1)
            cv2.imwrite(os.path.join("/media/cvpr/CM_1/COREMAX/pytesseract/", file_Name), img_bgr_rgb)
            print("finished")
