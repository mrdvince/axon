import cv2
import matplotlib.pyplot as plt
import numpy as np

img = cv2.imread('road.jpg')
## convert to rgb
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

height = img.shape[0]
width = img.shape[1]

region_of_interest_vertices = [
    (0, height),
    (width/2, height/2),
    (width, height)
]


def region_of_interest(img, vertices):
    mask = np.zeros_like(img)
    channel_count = img.shape[2]
    match_mask_color = (255,) * channel_count
    cv2.fillPoly(mask, vertices, match_mask_color)
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image

cropped_image = region_of_interest(img,
                np.array([region_of_interest_vertices], np.int32),)

plt.imshow(cropped_image)
plt.show()
