import argparse
from pathlib import Path
from tqdm import tqdm
from sleap_utils import Sleap_Tracks  # Assuming your class is in sleap_utils.py


def main():
    parser = argparse.ArgumentParser(
        description="Generate annotated videos from SLEAP tracking data."
    )
    parser.add_argument(
        "filenames", nargs="+", type=Path, help="Path(s) to the SLEAP tracking file(s)."
    )
    parser.add_argument(
        "--save",
        action="store_true",
        help="Whether to save the annotated video to a file.",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        help="Path to save the annotated video if save is True.",
    )
    parser.add_argument("--start", type=int, help="Starting frame number.")
    parser.add_argument("--end", type=int, help="Ending frame number.")
    parser.add_argument(
        "--nodes",
        type=str,
        nargs="+",
        help="Node name or list of node names to annotate.",
    )
    parser.add_argument(
        "--labels", action="store_true", help="Whether to display labels on the nodes."
    )
    parser.add_argument(
        "--edges", action="store_true", help="Whether to draw edges between nodes."
    )

    args = parser.parse_args()

    for filename in tqdm(args.filenames, desc="Processing files"):
        sleap_tracks = Sleap_Tracks(filename)
        sleap_tracks.generate_annotated_video(
            save=args.save,
            output_path=args.output_path,
            start=args.start,
            end=args.end,
            nodes=args.nodes,
            labels=args.labels,
            edges=args.edges,
        )


if __name__ == "__main__":
    main()
