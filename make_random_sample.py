import argparse
import random
from pathlib import Path
from tqdm import tqdm
from sleap_utils import Sleap_Tracks  # Assuming your class is in sleap_utils.py


def list_videos(folder):
    """Recursively list all video files in the given folder."""
    return [p for p in folder.rglob("*") if p.suffix in {".mp4", ".avi", ".mov"}]


def main():
    parser = argparse.ArgumentParser(
        description="Generate annotated videos from SLEAP tracking data in a folder."
    )
    parser.add_argument(
        "folder", type=Path, help="Path to the folder containing video files."
    )
    parser.add_argument(
        "--sample_size",
        type=int,
        default=None,
        help="Number of random videos to process. If not specified, process all videos.",
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

    # List all video files in the folder
    all_videos = list_videos(args.folder)

    # Select a random sample if sample_size is specified
    if args.sample_size is not None:
        videos_to_process = random.sample(
            all_videos, min(args.sample_size, len(all_videos))
        )
    else:
        videos_to_process = all_videos

    # Ensure the output directory exists
    if args.save and args.output_path:
        output_dir = Path(args.output_path)
        output_dir.mkdir(parents=True, exist_ok=True)

    for video in tqdm(videos_to_process, desc="Processing videos"):

        print(f"Processing video {video}")

        # Find the .h5 file that has "full" in its name and has the same parent directory as the video
        h5_files = list(video.parent.glob("*full*.h5"))
        if not h5_files:
            print(f"No matching .h5 file found for video {video}")
            continue
        h5_file = h5_files[0]

        sleap_tracks = Sleap_Tracks(h5_file)
        output_file = None
        if args.save and args.output_path:
            output_file = output_dir / f"{video.stem}_annotated.mp4"
        sleap_tracks.generate_annotated_video(
            save=args.save,
            output_path=str(output_file) if output_file else None,
            start=args.start,
            end=args.end,
            nodes=args.nodes,
            labels=args.labels,
            edges=args.edges,
        )


if __name__ == "__main__":
    main()
