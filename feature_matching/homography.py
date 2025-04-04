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

# Normalize function to map values to a given range
def normalize(value, min_val, max_val, target_min, target_max):
    value = np.clip(value, min_val, max_val)
    return target_min + (value - min_val) / (max_val - min_val) * (target_max - target_min)

def compute_combined_dynamic_ratio(gray1, gray2, keypoints1, keypoints2, motion_vectors=None, inlier_ratio=None):
    # 1. Scene Texture
    texture1 = cv2.Laplacian(gray1, cv2.CV_64F).var()
    texture2 = cv2.Laplacian(gray2, cv2.CV_64F).var()
    avg_texture = (texture1 + texture2) / 2
    texture_ratio = normalize(avg_texture, 10, 60, 0.8, 0.6)    # < 10:  # Low texture, > 50:  # High texture

    # 2. Motion Magnitude
    if motion_vectors is not None:
        avg_magnitude = np.mean(motion_vectors)
        motion_ratio = normalize(avg_magnitude, 5, 30, 0.8, 0.7)  # < 5:  # Small motion, > 30:  # Large motion
    else:
        motion_ratio = 0.75  # Default value if motion not provided

    # 3. Keypoint Density
    numPixels = len(gray1) * len(gray1[0])
    numKeypoints = len(keypoints1) + len(keypoints2) / 2
    kp_density = numKeypoints / numPixels
    keypoint_ratio = normalize(kp_density, 0.0001, 0.003, 0.8, 0.6)     # < 0.0001: Less keypoints, > 0.003: rich keypoints

    # 4. Inlier Ratio
    if inlier_ratio is not None:
        inlier_based_ratio = normalize(inlier_ratio, 0.3, 0.7, 0.8, 0.6)
    else:
        inlier_based_ratio = 0.75  # Default value if inlier ratio not available

    # Weighted combination
    #combined_ratio = (0.3 * texture_ratio +
    #                  0.3 * motion_ratio +
    #                  0.2 * keypoint_ratio +
    #                  0.2 * inlier_based_ratio)
    print(texture_ratio)
    print(keypoint_ratio)
    combined_ratio = (texture_ratio + keypoint_ratio) / 2
    
    return combined_ratio

abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)

# Read two consecutive frames
#frame1 = cv2.imread('../resource/frame_0014.png') 
#frame2 = cv2.imread('../resource/frame_0015.png') 

#frame1 = cv2.imread('../resource/local_056.jpg') 
#frame2 = cv2.imread('../resource/local_057.jpg') 

#frame1 = cv2.imread('../resource/img_036.jpg') 
#frame2 = cv2.imread('../resource/img_037.jpg') 

frame1 = cv2.imread('../resource/9I_Kpjcdw48/frame_002274.jpg') 
frame2 = cv2.imread('../resource/9I_Kpjcdw48/frame_002275.jpg') 

#frame1 = cv2.imread('../resource/B1xNOOJmAu8/frame_008754.jpg') 
#frame2 = cv2.imread('../resource/B1xNOOJmAu8/frame_008755.jpg') 

#frame1 = cv2.imread('../resource/MszwdOmEiPk/frame_000380.jpg') 
#frame2 = cv2.imread('../resource/MszwdOmEiPk/frame_000381.jpg') 

#frame1 = cv2.imread('../resource/vQ7jmoOKb6w/frame_024599.jpg') 
#frame2 = cv2.imread('../resource/vQ7jmoOKb6w/frame_024600.jpg') 

#frame1 = cv2.imread('../resource/Y8lChqjsV10/frame_001302.jpg') 
#frame2 = cv2.imread('../resource/Y8lChqjsV10/frame_001303.jpg') 

# Convert images to grayscale
gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

# Detect ORB features and compute descriptors
orb = cv2.ORB_create(nfeatures=5000)
keypoints1, descriptors1 = orb.detectAndCompute(gray1, None)
keypoints2, descriptors2 = orb.detectAndCompute(gray2, None)

# Match features using Brute Force Matcher
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
matches = bf.knnMatch(descriptors1, descriptors2, k=2)

# Dynamically determine Lowe's ratio
combined_ratio = compute_combined_dynamic_ratio(gray1, gray2, keypoints1, keypoints2)
print(combined_ratio)

# Apply Lowe's ratio test to remove poor matches
good_matches = []
for m, n in matches:
    if m.distance < combined_ratio * n.distance:
        good_matches.append(m)

# Extract matched points
points1 = np.float32([keypoints1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
points2 = np.float32([keypoints2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

# Estimate homography matrix using RANSAC (removes outliers)
H, mask = cv2.findHomography(points2, points1, cv2.RANSAC, 5.0)
print(H)

# Filter inlier matches based on RANSAC result
inlier_matches = [m for i, m in enumerate(good_matches) if mask[i]]

print(len(good_matches))
print(len(inlier_matches))
print(len(inlier_matches) / len(good_matches))

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
