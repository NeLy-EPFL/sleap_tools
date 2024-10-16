import h5py

import pandas as pd

import cv2

# import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from pathlib import Path

from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import ProcessPoolExecutor


class Sleap_Tracks:
    """Class for handling SLEAP tracking data. It is a wrapper around the SLEAP H5 file format."""

    class Animal:
        """Nested class to represent an animal with node properties."""

        def __init__(self, animal_data, node_names):
            self.node_names = node_names
            self.animal_data = animal_data

            for node in node_names:
                setattr(self, node, self._create_node_property(node))

        def _create_node_property(self, node):
            """Creates a property for a node to access its x and y coordinates for each frame.

            Args:
                node (str): The name of the node.

            Returns:
                property: A property for the node coordinates.
            """

            @property
            def node_property(self):
                x_values = self.animal_data[f"x_{node}"].values
                y_values = self.animal_data[f"y_{node}"].values
                return list(zip(x_values, y_values))

            return node_property

    def __init__(self, filename):
        """Initialize the Sleap_Track object with the given SLEAP tracking file.

        Args:
            filename (Path): Path to the SLEAP tracking file.
        """

        self.path = filename

        # Open the SLEAP tracking file
        self.h5file = h5py.File(filename, "r")

        self.node_names = [x.decode("utf-8") for x in self.h5file["node_names"]]
        self.edge_names = [
            [y.decode("utf-8") for y in x] for x in self.h5file["edge_names"]
        ]
        self.edges_idx = self.h5file["edge_inds"]

        self.tracks = self.h5file["tracks"]

        self.dataset = self.generate_tracks_data()

        self.video = Path(self.h5file["video_path"][()].decode("utf-8"))

        # Try to load the video file to check its accessibility
        try:
            cap = cv2.VideoCapture(str(self.video))
            cap.release()
        except:
            print(
                f"Video file not available: {self.video}. Check path and server access."
            )

        # Create animal properties
        self.animals = []
        for i in range(len(self.tracks)):
            animal_data = self.dataset[self.dataset["animal"] == f"animal_{i+1}"]
            self.animals.append(self.Animal(animal_data, self.node_names))

        print(f"Loaded SLEAP tracking file: {filename}")
        print(f"N° of animals: {len(self.tracks)}")
        print(f"Nodes: {self.node_names}")

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

            for k, n in enumerate(self.node_names):
                tracking_df[f"x_{n}"] = x_coords[k]
                tracking_df[f"y_{n}"] = y_coords[k]

            df_list.append(tracking_df)

        df = pd.concat(df_list, ignore_index=True)

        return df

    def generate_annotated_frame(self, frame, nodes=None, labels=False, edges=True):
        """Generates an annotated frame image for a specific frame."""
        frame_data = self.dataset[self.dataset["frame"] == frame]

        cap = cv2.VideoCapture(str(self.video))
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame - 1)

        ret, img = cap.read()
        if not ret:
            raise ValueError(f"Could not read frame {frame} from video {self.video}")

        if nodes is None:
            nodes = self.node_names
        elif isinstance(nodes, str):
            nodes = [nodes]

        for _, row in frame_data.iterrows():
            for node in nodes:
                x = row[f"x_{node}"]
                y = row[f"y_{node}"]
                if not np.isnan(x) and not np.isnan(y):
                    x, y = int(x), int(y)
                    cv2.circle(img, (x, y), 2, (0, 255, 255), -1)
                    if labels:
                        cv2.putText(
                            img,
                            node,
                            (x, y),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5,
                            (255, 255, 255),
                            1,
                            cv2.LINE_AA,
                        )

        if edges:
            for edge in self.edge_names:
                node1, node2 = edge
                if node1 in nodes and node2 in nodes:
                    x1 = frame_data[f"x_{node1}"].values[0]
                    y1 = frame_data[f"y_{node1}"].values[0]
                    x2 = frame_data[f"x_{node2}"].values[0]
                    y2 = frame_data[f"y_{node2}"].values[0]
                    if not (
                        np.isnan(x1) or np.isnan(y1) or np.isnan(x2) or np.isnan(y2)
                    ):
                        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                        cv2.line(img, (x1, y1), (x2, y2), (0, 255, 255), 1)

        cap.release()
        return img

    def generate_annotated_video(
        self,
        save=False,
        output_path=None,
        start=None,
        end=None,
        nodes=None,
        labels=False,
        edges=True,
    ):
        cap = cv2.VideoCapture(str(self.video))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if start is None:
            start = 1
        if end is None:
            end = total_frames

        if save:
            if output_path is None:
                output_path = self.video.with_suffix(".annotated.mp4")
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

        def process_frame(frame):
            return self.generate_annotated_frame(
                frame, nodes=nodes, labels=labels, edges=edges
            )

        with ThreadPoolExecutor() as executor:
            annotated_frames = list(
                tqdm(
                    executor.map(process_frame, range(start, end + 1)),
                    total=end - start + 1,
                    desc="Processing frames",
                )
            )

        for annotated_frame in annotated_frames:
            if save:
                out.write(annotated_frame)
            else:
                cv2.imshow("Annotated Video", annotated_frame)
                if cv2.waitKey(25) & 0xFF == ord("q"):
                    break

        cap.release()
        if save:
            out.release()
        else:
            cv2.destroyAllWindows()

    def gpu_generate_annotated_video(
        self,
        save=False,
        output_path=None,
        start=None,
        end=None,
        nodes=None,
        labels=False,
        edges=True,
    ):
        """Generates a video with GPU-accelerated annotated frames."""

        cap = cv2.VideoCapture(str(self.video))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if start is None:
            start = 1
        if end is None:
            end = total_frames

        if save:
            if output_path is None:
                output_path = self.video.with_suffix(".annotated.mp4")
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

        while cap.isOpened():
            ret, img = cap.read()
            if not ret or cap.get(cv2.CAP_PROP_POS_FRAMES) > end:
                break

            # Upload frame to GPU
            gpu_img = cv2.cuda_GpuMat()
            gpu_img.upload(img)

            frame = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
            frame_data = self.dataset[self.dataset["frame"] == frame]

            if nodes is None:
                nodes = self.node_names

            # Annotate nodes and labels
            for _, row in frame_data.iterrows():
                for node in nodes:
                    x = row[f"x_{node}"]
                    y = row[f"y_{node}"]
                    if not np.isnan(x) and not np.isnan(y):
                        x, y = int(x), int(y)
                        img = gpu_img.download()  # Download to CPU for annotations
                        cv2.circle(img, (x, y), 2, (0, 255, 255), -1)
                        if labels:
                            cv2.putText(
                                img,
                                node,
                                (x, y),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.5,
                                (255, 255, 255),
                                1,
                                cv2.LINE_AA,
                            )
                        gpu_img.upload(img)  # Upload back to GPU

            # Annotate edges if needed
            if edges:
                for edge in self.edge_names:
                    node1, node2 = edge
                    if node1 in nodes and node2 in nodes:
                        x1 = frame_data[f"x_{node1}"].values[0]
                        y1 = frame_data[f"y_{node1}"].values[0]
                        x2 = frame_data[f"x_{node2}"].values[0]
                        y2 = frame_data[f"y_{node2}"].values[0]
                        if not (
                            np.isnan(x1) or np.isnan(y1) or np.isnan(x2) or np.isnan(y2)
                        ):
                            img = gpu_img.download()  # CPU operations for drawing
                            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                            cv2.line(img, (x1, y1), (x2, y2), (0, 255, 255), 1)
                            gpu_img.upload(img)  # Back to GPU

            if save:
                out.write(gpu_img.download())
            else:
                cv2.imshow("Annotated Video", gpu_img.download())
                if cv2.waitKey(25) & 0xFF == ord("q"):
                    break

        cap.release()
        if save:
            out.release()
        else:
            cv2.destroyAllWindows()
