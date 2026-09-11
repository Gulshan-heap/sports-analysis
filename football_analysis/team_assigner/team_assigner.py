import numpy as np
import cv2
from sklearn.cluster import KMeans

class TeamAssigner:
    # Clustering on crops larger than this wastes time without improving the
    # 2-colour split; crops are downscaled to this height first.
    MAX_CROP_HEIGHT = 48

    def __init__(self):
        self.team_colors = {}
        self.player_team_dict = {}
        self.kmeans = None
    
    def get_clustering_model(self,image):
        # Reshape the image to 2D array
        image_2d = image.reshape(-1,3)

        # Preform K-means with 2 clusters (fixed seed so team colours are
        # reproducible across runs; KMeans labels are otherwise arbitrary)
        kmeans = KMeans(n_clusters=2, init="k-means++", n_init=1, random_state=0)
        kmeans.fit(image_2d)

        return kmeans

    def get_player_color(self,frame,bbox):
        image = frame[int(bbox[1]):int(bbox[3]),int(bbox[0]):int(bbox[2])]
        if image.size == 0 or image.shape[0] < 2:
            return np.array([128, 128, 128], dtype=np.float64)

        top_half_image = image[0:int(image.shape[0]/2),:]

        # Downscale tall crops so KMeans runs on far fewer pixels
        # (identical dominant-colour result, ~10x faster on 1080p frames).
        if top_half_image.shape[0] > self.MAX_CROP_HEIGHT:
            scale = self.MAX_CROP_HEIGHT / float(top_half_image.shape[0])
            top_half_image = cv2.resize(
                top_half_image, (0, 0), fx=scale, fy=scale,
                interpolation=cv2.INTER_AREA)

        # Get Clustering model
        kmeans = self.get_clustering_model(top_half_image)

        # Get the cluster labels forr each pixel
        labels = kmeans.labels_

        # Reshape the labels to the image shape
        clustered_image = labels.reshape(top_half_image.shape[0],top_half_image.shape[1])

        # Get the player cluster
        corner_clusters = [clustered_image[0,0],clustered_image[0,-1],clustered_image[-1,0],clustered_image[-1,-1]]
        non_player_cluster = max(set(corner_clusters),key=corner_clusters.count)
        player_cluster = 1 - non_player_cluster

        player_color = kmeans.cluster_centers_[player_cluster]

        return player_color


    def assign_team_color(self, frame, player_detections):
        
        player_colors = []
        for _, player_detection in player_detections.items():
            bbox = player_detection["bbox"]
            player_color =  self.get_player_color(frame,bbox)
            player_colors.append(player_color)
        
        if len(player_colors) < 2:
            # Only one team visible in this clip -> assign everyone to team 1.
            self.kmeans = None
            self.team_colors[1] = np.mean(player_colors, axis=0) if player_colors else np.array([128, 128, 128])
            self.team_colors[2] = self.team_colors[1].copy()
            return

        kmeans = KMeans(n_clusters=2, init="k-means++",n_init=10)
        kmeans.fit(player_colors)

        self.kmeans = kmeans

        self.team_colors[1] = kmeans.cluster_centers_[0]
        self.team_colors[2] = kmeans.cluster_centers_[1]


    def get_player_team(self, frame, player_bbox, player_id):
        if player_id in self.player_team_dict:
            return self.player_team_dict[player_id]

        player_color = self.get_player_color(frame, player_bbox)

        if self.kmeans is None:
            team_id = 1
        else:
            team_id = self.kmeans.predict(player_color.reshape(1, -1))[0]
            team_id += 1

        self.player_team_dict[player_id] = team_id

        return team_id