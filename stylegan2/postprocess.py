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
orgPath = '/data/suparna/Data/soccershirt/schalke_ready/42.png'
densePath = '/data/suparna/Data/soccershirt/schalke_densepose/42_mask.png'
pgn_mask = '/data/suparna/Data/soccershirt/schalke_voguemask/42.png'
predPath = 'out/24.png'

mask = Image.open(densePath)
mask = np.array(mask)

img = Image.open(orgPath)
img = np.array(img)
imgcopy = img.copy()
pred = Image.open(predPath)

#w, h, _  = img.shape

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
maskx = np.any(mask_t, axis=0)
masky = np.any(mask_t, axis=1)
x1 = np.argmax(maskx)
y1 = np.argmax(masky)
x2 = len(maskx) - np.argmax(maskx[::-1])
y2 = len(masky) - np.argmax(masky[::-1])
#print(x1, y1, x2, y2)

# wa, ha = 256/(y2-y1), 256/(x2-x1)
# w1, h1 = (y1-0)*wa , (x1-0)*ha
# print(w1,h1)
# w2, h2 = (w-y2)*wa , (h-x2)*ha
# print(w2,h2)
# sub1 = Image.fromarray(img[0:y1, 0:x1]).resize(w1, h1)
# sub1 = Image.fromarray(img[0:y1, 0:x1]).resize(w2, h2)
# empty = np.ones((256,h2-h1))

sub_image = img[y1:y2, x1:x2]
print(sub_image.shape)
h, w, c = sub_image.shape
pred = pred.resize((w,h))
img[y1:y2, x1:x2] = pred


LIPTOVOGUE = {
    'background': [0],
    'tops': [5,6,7,11], 
    'bottoms': [8,9,10,12], 
    'face': [13], 
    'hair': [1,2], 
    'arms': [3, 14, 15], 
    'skin': [85,51,0], 
    'legs': [16,17,18,19], 
    'other':[]
}
colorspace = [(255, 255, 255), (255, 85, 0), (85, 85, 0), (0, 0, 255), (0, 119, 221), (51, 170, 221), (85,51,0), (170, 255, 85), (52, 86, 128)]
print(colorspace[0])
im_parse_rgb = np.array(Image.open(pgn_mask))    
    
for j, key in enumerate(LIPTOVOGUE.keys()):
    if key in ['tops']: continue
    img[(im_parse_rgb == colorspace[j]).all(-1)] = imgcopy[(im_parse_rgb == colorspace[j]).all(-1)]



pred_full = Image.fromarray(img.astype(np.uint8))
pred_full.save('out/42_full.png')