import csv
import numpy as np
import matplotlib
matplotlib.use("Agg")  # headless-safe backend; prevents GUI crashes on Windows
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde


def export_frame_data_csv(tracks, team_ball_control, output_path="output_videos/match_data.csv"):
    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["frame", "player_id", "team", "x", "y", "speed", "distance", "has_ball", "team_in_control"])
        for frame_num, player_track in enumerate(tracks['players']):
            control = team_ball_control[frame_num] if frame_num < len(team_ball_control) else None
            for player_id, info in player_track.items():
                pos = info.get('position_transformed')
                x, y = (pos if pos else (None, None))
                writer.writerow([
                    frame_num, player_id, info.get('team'),
                    x, y, info.get('speed'), info.get('distance'),
                    info.get('has_ball', False), control
                ])
    print(f"CSV exported to {output_path}")


def generate_player_heatmap(tracks, player_id, court_length=23.32, court_width=68,
                             output_path=None):
    xs, ys = [], []
    for frame in tracks['players']:
        if player_id in frame:
            pos = frame[player_id].get('position_transformed')
            if pos:
                xs.append(float(pos[0]))
                ys.append(float(pos[1]))

    if not xs:
        print(f"No position data found for player {player_id}")
        return

    plt.figure(figsize=(10, 7))

    # KDE via scipy + matplotlib. (seaborn.kdeplot triggered native crashes in
    # pandas/pyarrow when the ML stack was loaded; this path avoids it.)
    kde_drawn = False
    if len(xs) >= 2:
        try:
            kde = gaussian_kde(np.vstack([xs, ys]))
            grid_x = np.linspace(0, court_length, 120)
            grid_y = np.linspace(0, court_width, 120)
            gx, gy = np.meshgrid(grid_x, grid_y)
            zi = kde(np.vstack([gx.ravel(), gy.ravel()])).reshape(gx.shape)
            plt.pcolormesh(gx, gy, zi, cmap="Reds", shading="auto")
            kde_drawn = True
        except np.linalg.LinAlgError:
            pass  # all points identical etc. -> fall back to scatter
    if not kde_drawn:
        plt.scatter(xs, ys, c="red", s=40, alpha=0.6)

    plt.xlim(0, court_length)
    plt.ylim(0, court_width)
    plt.gca().invert_yaxis()
    plt.title(f"Heatmap — Player {player_id}")
    if output_path:
        plt.savefig(output_path)
        print(f"Saved heatmap to {output_path}")
    else:
        plt.show()
    plt.close()


def team_possession_summary(team_ball_control):
    team_ball_control = np.array(team_ball_control)
    total = len(team_ball_control)
    for team in np.unique(team_ball_control):
        pct = (team_ball_control == team).sum() / total * 100
        print(f"Team {team}: {pct:.1f}% possession")