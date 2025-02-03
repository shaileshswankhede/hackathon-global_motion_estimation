import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

def show_image(title, img):
    plt.figure(figsize=(10, 6))
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.imshow(img, cmap='gray')
    plt.title(title)
    plt.axis('off')
    #plt.show()

def compute_psnr(original, compensated):
    mse = np.mean((original - compensated) ** 2)
    return 20 * np.log10(255.0 / np.sqrt(mse))

abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)

# Read two consecutive frames
frame1 = cv2.imread('../resource/frame_0014.png')  # Replace with your file path
frame2 = cv2.imread('../resource/frame_0015.png')  # Replace with your file path

# Convert images to grayscale
gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

# Detect ORB features and compute descriptors
orb = cv2.ORB_create(500)
keypoints1, descriptors1 = orb.detectAndCompute(gray1, None)
keypoints2, descriptors2 = orb.detectAndCompute(gray2, None)

# Match features using Brute Force Matcher
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
matches = bf.knnMatch(descriptors1, descriptors2, k=2)

# Apply Lowe's ratio test to remove poor matches
good_matches = []
for m, n in matches:
    if m.distance < 0.75 * n.distance:
        good_matches.append(m)

# Extract matched points
points1 = np.float32([keypoints1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
points2 = np.float32([keypoints2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

# Estimate homography matrix using RANSAC (removes outliers)
H, mask = cv2.findHomography(points2, points1, cv2.RANSAC, 5.0)
print(H)

# Filter inlier matches based on RANSAC result
inlier_matches = [m for i, m in enumerate(good_matches) if mask[i]]

# Draw inlier matches
inlier_img = cv2.drawMatches(frame1, keypoints1, frame2, keypoints2, inlier_matches, None, flags=2)
show_image("Inlier Matches After RANSAC", inlier_img)

# Warp the second frame to align with the first frame
height, width = gray1.shape
aligned_frame2 = cv2.warpPerspective(frame2, H, (width, height))
diff = abs(frame2 - aligned_frame2)
#diff = abs(gray2 - cv2.cvtColor(aligned_frame2, cv2.COLOR_BGR2GRAY))

# Display results
show_image("Original Frame 1", frame1)
show_image("Original Frame 2", frame2)
show_image("Motion-Compensated Frame 2", aligned_frame2)
show_image("diff", diff)
psnr = compute_psnr(aligned_frame2, frame2)
print(psnr)
plt.show()
