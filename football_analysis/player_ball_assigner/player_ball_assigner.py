import sys 
sys.path.append('../')
from utils import get_center_of_bbox, measure_distance

class PlayerBallAssigner():
    # Reference distance tuned for a 1080p frame; scaled for other resolutions.
    REFERENCE_HEIGHT = 1080
    REFERENCE_DISTANCE = 70

    def __init__(self, frame_height=1080):
        self.frame_height = frame_height
        self.set_frame_height(frame_height)

    def set_frame_height(self, frame_height):
        self.frame_height = frame_height or self.REFERENCE_HEIGHT
        self.max_player_ball_distance = int(
            self.REFERENCE_DISTANCE * self.frame_height / self.REFERENCE_HEIGHT
        )
    
    def assign_ball_to_player(self,players,ball_bbox):
        ball_position = get_center_of_bbox(ball_bbox)

        miniumum_distance = 99999
        assigned_player=-1

        for player_id, player in players.items():
            player_bbox = player['bbox']

            distance_left = measure_distance((player_bbox[0],player_bbox[-1]),ball_position)
            distance_right = measure_distance((player_bbox[2],player_bbox[-1]),ball_position)
            distance = min(distance_left,distance_right)

            if distance < self.max_player_ball_distance:
                if distance < miniumum_distance:
                    miniumum_distance = distance
                    assigned_player = player_id

        return assigned_player