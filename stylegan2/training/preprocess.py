import numpy as np
from PIL import Image
import os

'''
Densepose labels
0      = Background
1, 2   = Torso
3      = Right Hand
4      = Left Hand
5      = Right Foot
6      = Left Foot
7, 9   = Upper Leg Right
8, 10  = Upper Leg Left
11, 13 = Lower Leg Right
12, 14 = Lower Leg Left
15, 17 = Upper Arm Left
16, 18 = Upper Arm Right
19, 21 = Lower Arm Left
20, 22 = Lower Arm Right
23, 24 = Head
'''
src = '/data/suparna/Data/soccershirt/schalke_small'
src_mask = '/data/suparna/Data/soccershirt/schalke_densepose'
pgn_mask = '/data/suparna/Data/soccershirt/schalke_mask'
dest = '/data/suparna/Data/soccershirt/schalke_new'
for im in os.listdir(src):
    ###### read iuv image
    print(os.path.join(src,im))
    mask = Image.open(os.path.join(src_mask,im.replace('.png','_mask.png')))
    mask = np.array(mask)
    img = Image.open(os.path.join(src,im))
    img = np.array(img)

    iuv = Image.open(os.path.join(src_mask, im))
    iuv = np.array(iuv)
    #cloth = Image.open(os.path.join(src, folder, 'cloth.jpg'))
    mask_1 = np.where(mask == 1,1,0)
    mask_2 = np.where(mask == 2,1,0)
    mask_3 = np.where(mask == 15,1,0)
    mask_4 = np.where(mask == 17,1,0)
    mask_5 = np.where(mask == 16,1,0)
    mask_6 = np.where(mask == 18,1,0)
    mask_7 = np.where(mask == 19,1,0)
    mask_8 = np.where(mask == 21,1,0)
    mask_9 = np.where(mask == 20,1,0)
    mask_10 = np.where(mask == 22,1,0)
    mask_t = mask_1 + mask_2 + mask_3 + mask_4 +mask_5 + mask_6 + mask_7 + mask_8 + mask_9 + mask_10

    iuv[(iuv == (0,0,0)).all(-1)] = (255,255,255)
    ########## detect bbox from mask
    maskx = np.any(mask_t, axis=0)
    masky = np.any(mask_t, axis=1)
    x1 = np.argmax(maskx)
    y1 = np.argmax(masky)
    x2 = len(maskx) - np.argmax(maskx[::-1])
    y2 = len(masky) - np.argmax(masky[::-1])
    sub_image = img[y1:y2, x1:x2]
    sub_image = Image.fromarray(sub_image.astype(np.uint8)).resize((256,256))
    sub_iuv = iuv[y1:y2, x1:x2]
    sub_iuv = Image.fromarray(sub_iuv).resize((256,256))
    #cloth = cloth.resize((256,256))

    sub_image.save(os.path.join(dest,im))
    sub_iuv.save(os.path.join(dest,im.replace('.png','_dense.png')))
    #cloth.save(os.path.join(src,folder,'test_cloth.png'))
    
