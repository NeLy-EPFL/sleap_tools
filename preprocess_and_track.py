from preprocess_video import process_videos_in_directory
from pathlib import Path
import subprocess

# Process all videos in the directory
input_dir = Path(
    "/mnt/upramdya_data/MD/MultiMazeRecorder/Videos/231115_TNT_Fine_1_Videos_Tracked/"
)

process_videos_in_directory(input_dir, exclude_keywords=["_preprocessed", "_annotated"])

# Path to your SLEAP model
model_path = "/mnt/upramdya_data/_Tracking_models/Sleap/mazerecorder/FlyTracking/FullBody/models/240910_140844.single_instance.n=421"


def run_sleap_on_preprocessed_videos(
    preprocessed_dir, model_path, make_annotated_video=True
):
    # Iterate over the preprocessed videos
    for video_path in Path(preprocessed_dir).rglob("*_preprocessed.mp4"):
        print(f"Running SLEAP on video: {video_path}")

        # Add "_full_body" to the video filename before the .slp extension
        slp_name = video_path.with_name(video_path.stem + "_full_body").with_suffix(
            ".slp"
        )

        # Check if the .slp file already exists
        if slp_name.exists():
            print(
                f"Skipping SLEAP tracking for {video_path}: {slp_name} already exists."
            )
        else:
            try:
                # Command to run SLEAP inference using the pre-trained model
                sleap_command = [
                    "sleap-track",
                    str(video_path),  # Use the video path as input
                    "--model",
                    model_path,
                    "--tracking.tracker",
                    "flow",
                    "--output",
                    str(slp_name),
                ]

                # Execute the SLEAP command to generate the .slp file
                subprocess.run(sleap_command, check=True)
                print(f"Completed SLEAP tracking for {video_path}")

            except subprocess.CalledProcessError as e:
                print(f"Error running SLEAP tracking for {video_path}: {e}")
                continue

        if slp_name.exists():
            h5_file = slp_name.with_suffix(".h5")  # Output h5 file

            if h5_file.exists():
                print(f"Skipping conversion for {slp_name}: {h5_file} already exists.")
            else:
                try:
                    # Command to convert .slp to .h5
                    sleap_convert_command = [
                        "sleap-convert",
                        str(slp_name),
                        "--format",
                        "analysis",
                        "--output",
                        str(h5_file),
                    ]

                    # Execute the conversion command
                    subprocess.run(sleap_convert_command, check=True)
                    print(f"Converted {slp_name} to {h5_file}")

                except subprocess.CalledProcessError as e:
                    print(f"Error converting {slp_name} to {h5_file}: {e}")
                    continue

            if make_annotated_video:
                annotated_video_path = video_path.with_name(
                    video_path.stem + "_annotated"
                ).with_suffix(".mp4")

                if annotated_video_path.exists():
                    print(
                        f"Skipping annotated video generation for {h5_file}: {annotated_video_path} already exists."
                    )
                else:
                    try:
                        # Now call makevideo.py to generate the annotated video
                        print(f"Generating annotated video for {h5_file}")
                        makevideo_command = [
                            "python3",
                            "makevideo.py",
                            str(h5_file),  # Pass the h5 file as input
                            "--save",  # Save the annotated video
                            "--output_path",
                            str(annotated_video_path),
                            "--edges",  # Draw edges between nodes
                        ]

                        # Execute the command to generate the video
                        subprocess.run(makevideo_command, check=True)
                        print(f"Generated annotated video for {h5_file}")

                    except subprocess.CalledProcessError as e:
                        print(f"Error generating annotated video for {h5_file}: {e}")

        else:
            print(f"Error: {slp_name} not found. SLEAP tracking might have failed.")


# Run SLEAP tracking, conversion, and video generation on all preprocessed videos
run_sleap_on_preprocessed_videos(input_dir, model_path)

print("All videos have been preprocessed, tracked, and annotated.")
