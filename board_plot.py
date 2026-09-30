import matplotlib.pyplot as plt
import compress_pickle
import random
import os
import glob


def load_data(filename):
    return compress_pickle.load(filename)


class Point:
    def __init__(self, x, y, timestamp, force=1):
        self.x = x
        self.y = y
        self.timestamp = timestamp
        self.force = force

    def __repr__(self):
        return f"Point(x={self.x}, y={self.y}, timestamp={self.timestamp})"


def visualize_touch_data(
        directory, start_time, end_time, name="raw", if_save=False, save_path=None
):
    files = glob.glob(os.path.join(directory, "*.gz"))
    fig, ax = plt.subplots()

    all_x_coords = []
    all_y_coords = []
    all_forces = []

    for file in files:
        try:
            data = load_data(file)
            filtered_data = [
                point for point in data if start_time <= point.timestamp <= end_time
            ]
            if filtered_data:
                for point in filtered_data:
                    x_coords = [contact.x for contact in point.contacts]
                    y_coords = [1 - contact.y for contact in point.contacts]
                    force = [contact.force for contact in point.contacts]

                    all_x_coords.extend(x_coords)
                    all_y_coords.extend(y_coords)
                    all_forces.extend(force)
        except Exception as e:
            print(f"Error loading {file}: {e}")

    if all_x_coords:
        sc = ax.scatter(all_x_coords, all_y_coords, c=all_forces, cmap='viridis', s=40)
        plt.colorbar(sc, label="Force")

    ax.set_xlabel("X Coordinate")
    ax.set_ylabel("Y Coordinate")
    ax.set_title(name)

    if if_save and save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
    else:
        plt.show()


def plot_board_data(data_dir, user):
    user_dir = os.path.join(data_dir, user)
    action_dirs = glob.glob(os.path.join(user_dir, "*"))
    print(action_dirs)
    for action_dir in action_dirs:
        timestamp_files = glob.glob(os.path.join(action_dir, "*_timestamp.txt"))
        if not timestamp_files:
            continue
        j = random.randint(0, len(timestamp_files) - 1)
        with open(timestamp_files[j], "r") as f:
            lines = f.readlines()
            timestamps = []
            name = []
            for line in lines:
                timestamps.append(int(line.split()[0]))
                name.append(line.split()[1])
            i = random.randint(0, len(timestamps) - 2)
            start_time = timestamps[i]
            end_time = timestamps[i + 1]
            if not os.path.exists(f"./images/board/{user}"):
                os.makedirs(f"./images/board/{user}")
            action = action_dir.split("\\")[-1]
            if j % 2 == 0:
                name = name[i]
                name = name.lower()
            else:
                name = name[i]
                name = name.upper()
            visualize_touch_data(
                action_dir,
                start_time,
                end_time,
                f"{user}_{action}_{name}",
                if_save=True,
                save_path=f"./images/board/{user}/{user}_{action}_{name}.png",
            )


if __name__ == "__main__":
    directory = "./data"
    user = "user_0"
    plot_board_data(directory, user)