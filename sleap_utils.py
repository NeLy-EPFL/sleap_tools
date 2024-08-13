import h5py

import pandas as pd


class Sleap_Track:
    """Class for handling SLEAP tracking data. It is a wrapper around the SLEAP H5 file format."""

    def __init__(self, filename):
        """Initialize the Sleap_Track object with the given SLEAP tracking file.

        Args:
            filename (Path): Path to the SLEAP tracking file.
        """

        self.path = filename

        # Open the SLEAP tracking file
        self.h5file = h5py.File(filename, "r")

        self.nodes = [x.decode("utf-8") for x in self.h5file["node_names"]]

        self.tracks = self.h5file["tracks"]

        self.data = self.generate_tracks_data()

        print(f"Loaded SLEAP tracking file: {filename}")

    def generate_tracks_data(self):
        """Generates a pandas DataFrame with the tracking data, with the following columns:
        - frame
        for each node:
        - node_x
        - node_y

        The shape of the tracks is instance, x or y, nodes and frame and the order of the nodes is the same as the one in the nodes attribute.

        Returns:
            DataFrame: DataFrame with the tracking data.
        """

        df_list = []

        for i, animal in enumerate(self.tracks):

            animal = self.tracks[i]

            x_coords = animal[0]

            y_coords = animal[1]

            frames = range(1, len(animal[0][0]) + 1)

            tracking_df = pd.DataFrame(frames, columns=["frame"])

            # Give each animal some number

            tracking_df["animal"] = f"animal_{i+1}"

            for k, n in enumerate(self.nodes):
                tracking_df[f"x_{n}"] = x_coords[k]
                tracking_df[f"y_{n}"] = y_coords[k]

            df_list.append(tracking_df)

        df = pd.concat(df_list, ignore_index=True)

        return df
