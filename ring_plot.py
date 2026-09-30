import numpy as np
import matplotlib.pyplot as plt
import os


def plot_ring_data(file_path="", if_save=True):
    data_ring0 = np.fromfile(file_path, dtype=np.float64)
    data_ring0 = data_ring0.reshape(-1, 7)
    timestamp_path = file_path.replace("ring_0.bin", "timestamp.txt")
    timestampes = []
    with open(timestamp_path, "r", encoding="utf-8") as f:
        for line in f.readlines():
            timestampes.append(float(line.split(" ")[0]))
    for i in range(len(timestampes)):
        timestamp = timestampes[i]
        closet_idx = np.argmin(np.abs(data_ring0[:, -1] - timestamp))
        timestampes[i] = closet_idx
    plt.figure(figsize=(20, 10))
    plt.subplot(111)
    plt.plot(data_ring0[:, 0], label="acc_x")
    plt.plot(data_ring0[:, 1], label="acc_y")
    plt.plot(data_ring0[:, 2], label="acc_z")
    plt.plot(data_ring0[:, 3], label="gyro_x")
    plt.plot(data_ring0[:, 4], label="gyro_y")
    plt.plot(data_ring0[:, 5], label="gyro_z")
    for timestamp in timestampes:
        plt.axvline(x=timestamp, color="red")
    plt.legend()
    plt.title("ring0")
    if if_save:
        savepath = file_path.replace(".bin", ".png").replace("./data/", "./images/ring/")
        if not os.path.exists(os.path.dirname(savepath)):
            os.makedirs(os.path.dirname(savepath))
        plt.savefig(savepath)
        plt.close()
    else:
        plt.show()


if __name__ == "__main__":
    plot_ring_data(file_path="./data/user_0/3/0_ring_0.bin")