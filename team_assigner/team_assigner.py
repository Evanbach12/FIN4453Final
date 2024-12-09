from sklearn.cluster import KMeans


# Team assigner based on kmeans clustering. Seeks to differentiate player by team acordint to jersey pixel colors

class TeamAssigner:
    def __init__(self):
        self.team_colors = {}
        self.Player_team_dict = {}
    
    def get_clustering_model(self,image):
        # Reshape the image to 2D array
        image_2d = image.reshape(-1,3)

        # Preform K-means with 2 clusters
        kmeans = KMeans(n_clusters=2, init="k-means++",n_init=1)
        kmeans.fit(image_2d)

        return kmeans

    def get_Player_color(self,frame,bbox):
        image = frame[int(bbox[1]):int(bbox[3]),int(bbox[0]):int(bbox[2])]

        top_half_image = image[0:int(image.shape[0]/2),:]

        kmeans = self.get_clustering_model(top_half_image)

    
        labels = kmeans.labels_

        clustered_image = labels.reshape(top_half_image.shape[0],top_half_image.shape[1])

        corner_clusters = [clustered_image[0,0],clustered_image[0,-1],clustered_image[-1,0],clustered_image[-1,-1]]
        non_Player_cluster = max(set(corner_clusters),key=corner_clusters.count)
        Player_cluster = 1 - non_Player_cluster

        Player_color = kmeans.cluster_centers_[Player_cluster]

        return Player_color


    def assign_team_color(self,frame, Player_detections):
        
        Player_colors = []
        for _, Player_detection in Player_detections.items():
            bbox = Player_detection["bbox"]
            Player_color =  self.get_Player_color(frame,bbox)
            Player_colors.append(Player_color)
        
        kmeans = KMeans(n_clusters=2, init="k-means++",n_init=10)
        kmeans.fit(Player_colors)

        self.kmeans = kmeans

        self.team_colors[1] = kmeans.cluster_centers_[0]
        self.team_colors[2] = kmeans.cluster_centers_[1]


    def get_Player_team(self,frame,Player_bbox,Player_id):
        if Player_id in self.Player_team_dict:
            return self.Player_team_dict[Player_id]

        Player_color = self.get_Player_color(frame,Player_bbox)

        team_id = self.kmeans.predict(Player_color.reshape(1,-1))[0]
        team_id+=1

        if Player_id ==91:
            team_id=1

        self.Player_team_dict[Player_id] = team_id

        return team_id