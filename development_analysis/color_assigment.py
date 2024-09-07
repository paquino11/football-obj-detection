import cv2
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans


image_path = "../output_videos/cropped_image.jpg"
image = cv2.imread(image_path)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

plt.imshow(image)
plt.show()

#take the top half of the image
top_half_image = image[0: int(image.shape[0]/2), :]
plt.imshow(top_half_image)
plt.show()

# cluster the image into two clusters
# reshape the image into a 2d array
image_2d = top_half_image.reshape(-1, 3)

#perform k-means clustering
kmeans = KMeans(n_clusters=2, random_state=0).fit(image_2d)

#get the cluster labels
labels = kmeans.labels_

# reshape the labels into the original image shape
clustered_image = labels.reshape(top_half_image.shape[0], top_half_image.shape[1])

#display the clustered image
plt.imshow(clustered_image)
plt.show()

# make sure and assign the cluster ids are the same for all the frames
corner_clusters = [clustered_image[0,0], clustered_image[0,-1], clustered_image[-1,0], clustered_image[-1,-1]]
non_player_cluster = max(set(corner_clusters), key=corner_clusters.count)
print(non_player_cluster)

player_cluster = 1-non_player_cluster
print(player_cluster)

# color of the players shirt
kmeans.cluster_centers_[player_cluster]

