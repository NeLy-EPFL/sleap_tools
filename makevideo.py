import argparse
from pathlib import Path
from tqdm import tqdm
from sleap_utils import Sleap_Tracks  # Assuming your class is in sleap_utils.py


def main():
    """
    Main function to generate annotated videos from SLEAP tracking data.

    This script processes SLEAP tracking files and generates annotated videos
    with options to save the output, specify frame ranges, and annotate specific nodes.

    Usage:
        python makevideo.py <filenames> [--save] [--output_path <path>] [--start <frame>] [--end <frame>]
        [--nodes <node1> <node2> ...] [--labels] [--edges] [--gpu]

    Arguments:
        filenames (str): Path(s) to the SLEAP tracking file(s).
        --save (flag): Whether to save the annotated video to a file.
        --output_path (str): Path to save the annotated video if save is True.
        --start (int): Starting frame number.
        --end (int): Ending frame number.
        --nodes (str): Node name or list of node names to annotate.
        --labels (flag): Whether to display labels on the nodes.
        --edges (flag): Whether to draw edges between nodes.
        --gpu (flag): Whether to use the GPU-accelerated version of the annotation.
    """

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
    parser.add_argument(
        "--gpu", action="store_true", help="Use GPU acceleration for video processing."
    )

    args = parser.parse_args()

    print(f"Arguments: {args}")

    for filename in tqdm(args.filenames, desc="Processing files"):
        print(f"Processing file: {filename}")
        sleap_tracks = Sleap_Tracks(filename)

        if args.gpu:
            # Use the GPU version
            print("Using GPU for video processing...")
            sleap_tracks.gpu_generate_annotated_video(
                save=args.save,
                output_path=args.output_path,
                start=args.start,
                end=args.end,
                nodes=args.nodes,
                labels=args.labels,
                edges=args.edges,
            )
        else:
            # Use the CPU version
            print("Using CPU for video processing...")
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
