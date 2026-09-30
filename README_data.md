# WritingRing Dataset

---
license: cc
---

## Data Format and Structure

The data is organized under `./data/{username}/{action}`, where each action corresponds to a different handwriting task:

- **Action 0:** Letters, hand raised
- **Action 1:** Letters, hand not raised
- **Action 2:** Words (connected), hand raised
- **Action 3:** Words (connected), hand not raised
- **Action 4:** Words (disconnected), hand raised
- **Action 5:** Words (disconnected), hand not raised

Each action contains multiple data sets, consisting of data from a ring and a touchpad. Due to the size of the touchpad data, it's split into multiple parts. For example, for dataset with ID 1:

- Ring 0 data: `1_ring_0.bin`
- Touchpad data: `1_board_{index}.gz`
- Ring 1 data: `1_ring_1.bin` (Please ignore this.)

### Ring Data Format

The ring data is stored in numpy format as 64-bit floating point values. After reshaping with `data_ring0 = data_ring0.reshape(-1, 7)`, the columns represent:

1. Acceleration X
2. Acceleration Y
3. Acceleration Z
4. Angular velocity X
5. Angular velocity Y
6. Angular velocity Z
7. Timestamp

The data is sampled at 200Hz.

### Touchpad Data Format

The touchpad data is stored in gzipped format (`.gz`).

## Data Visualization

### Ring Data Visualization

The `ring_plot.py` script can be used to visualize the ring data. It contains a function `plot_ring_data()`, which takes two parameters:

1. Path to the ring data file
2. A flag indicating whether to save the output

The output is a plot that visualizes the ring data.

### Touchpad Data Visualization

The `board_plot.py` script is used for visualizing touchpad data. It contains a function `plot_board_data()`, which takes two parameters:

1. Path to the touchpad data file
2. Username

The function randomly samples touchpad data and outputs a visualization showing the user's writing trajectory, with color representing the pressure applied. Additionally, the `visualize_touch_data()` function can be called to visualize touchpad data for a specific time range.