import cv2
import matplotlib.pyplot as plt
import numpy as np

img_bgr = cv2.imread('image/lena.png')
img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

plt.figure(figsize=(12, 6))
plt.subplot(121)
plt.imshow(img_bgr)
plt.subplot(122)
plt.imshow(img_rgb)
plt.show()

img_hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)

# corner detection
img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
corners = cv2.cornerHarris(img_gray, 2, 3, 0.04)
plt.imshow(corners, cmap='gray')
plt.show()

sift = cv2.xfeatures2d.SIFT_create()
kp = sift.detect(img_bgr)
img_kp = np.zeros_like(img_bgr)
img_kp = cv2.drawKeypoints(img_rgb, kp, img_kp, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
plt.imshow(img_kp)
plt.show()

kp, des = sift.compute(img_bgr, kp)
print(des.shape)

kp2, des2 = sift.detectAndCompute(img_bgr, None)
print(np.allclose(des, des2))

surf = cv2.xfeatures2d.SURF_create()
kp = surf.detect(img_bgr)
img_kp = cv2.drawKeypoints(img_rgb, kp, img_kp, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
plt.imshow(img_kp)
plt.show()