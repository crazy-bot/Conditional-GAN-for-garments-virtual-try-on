import sys,os
sys.path.append("/pratika/pratika/code/styleposegan/detectron/DensePose_detectron2/")
sys.path.append("/pratika/pratika/code/styleposegan/detectron/")
from dense_pose_wrapper import *

import  glob
directory = '/pratika/suparna/Data/Zalando/'
for fol in os.listdir(directory):
    filename = directory + fol + '/0.jpg'
    print(filename)
    img = cv2.imread(filename)
    iuv_img = get_img_iuv_array(img)
    save_filename = directory + fol + '/0_dense.png'
    cv2.imwrite(save_filename, iuv_img.cpu().numpy() * 255)

