import time
import yaml
from preprocess_video import *
from pathlib import Path
import subprocess
import cv2
import sys

# Function to track the duration of each section
def timeit(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"Time taken for {func.__name__}: {end_time - start_time:.2f} seconds")
        return result

    return wrapper

# Wrapper function to run SLEAP commands in the `sleap_dev` environment
def run_in_sleap_env(command):
    """Run a command using the sleap_dev environment."""
    sleap_env_command = f"conda run -n sleap_dev {' '.join(command)}"
    result = subprocess.run(
        sleap_env_command, shell=True, capture_output=True, text=True
    )
    return result

# Path to your SLEAP model
model_path = "/mnt/upramdya_data/_Tracking_models/Sleap/mazerecorder/FlyTracking/FullBody/models/240910_140844.single_instance.n=421"

# Template size
template_width = 96
template_height = 516

@timeit
def prepare_videos(input_dir, use_gpu=True):
    """Preprocess videos in the directory."""
    process_videos_in_directory(
        input_dir, exclude_keywords=["_preprocessed", "_annotated"], use_gpu=use_gpu
    )

@timeit
def run_sleap_on_preprocessed_videos(
    preprocessed_dir, model_path, make_annotated_video=True, frame_batch_size=16
):
    """
    Run SLEAP on preprocessed videos and generate annotations.
    - Uses the `sleap_dev` environment for SLEAP commands.
    """
    for video_path in Path(preprocessed_dir).rglob("*_preprocessed.mp4"):
        print(f"Preparing video for SLEAP: {video_path}")
        slp_name = video_path.with_name(video_path.stem + "_full_body").with_suffix(
            ".slp"
        )

        # Skip if the SLEAP output already exists
        if slp_name.exists():
            print(f"Skipping {video_path}: {slp_name} already exists.")
            continue

        try:
            print(f"Running SLEAP on video: {video_path}")
            sleap_command = [
                "sleap-track",
                str(video_path),
                "--model",
                model_path,
                "--tracking.tracker",
                "flow",
                "--batch_size",
                str(frame_batch_size),
                "--output",
                str(slp_name),
            ]
            # Run SLEAP command in the sleap_dev environment
            result = run_in_sleap_env(sleap_command)
            print(f"SLEAP tracking completed with result: {result.stdout}")

        except subprocess.CalledProcessError as e:
            print(f"Error running SLEAP tracking for video: {e}")
            continue

    # Conversion and annotated video generation
    for video_path in Path(preprocessed_dir).rglob("*_preprocessed.mp4"):
        slp_name = video_path.with_name(video_path.stem + "_full_body").with_suffix(
            ".slp"
        )
        if not slp_name.exists():
            print(f"Error: {slp_name} not found. Skipping conversion.")
            continue

        h5_file = slp_name.with_suffix(".h5")
        if h5_file.exists():
            print(f"Skipping conversion for {slp_name}: {h5_file} already exists.")
        else:
            try:
                sleap_convert_command = [
                    "sleap-convert",
                    str(slp_name),
                    "--format",
                    "analysis",
                    "--output",
                    str(h5_file),
                ]
                # Convert .slp to .h5 using sleap_dev
                result = run_in_sleap_env(sleap_convert_command)
                print(f"Converted {slp_name} to {h5_file} with result: {result.stdout}")

            except subprocess.CalledProcessError as e:
                print(f"Error converting {slp_name} to {h5_file}: {e}")
                continue

        if make_annotated_video:
            annotated_video_path = video_path.with_name(
                video_path.stem + "_annotated"
            ).with_suffix(".mp4")
            if annotated_video_path.exists():
                print(
                    f"Skipping annotated video: {annotated_video_path} already exists."
                )
            else:
                try:
                    makevideo_command = [
                        "python3",
                        "makevideo.py",
                        str(h5_file),
                        "--save",
                        "--output_path",
                        str(annotated_video_path),
                        "--edges",
                        "--gpu",
                    ]
                    result = subprocess.run(makevideo_command, check=True)
                    print(
                        f"Generated annotated video: {annotated_video_path} with result: {result.stdout}"
                    )

                except subprocess.CalledProcessError as e:
                    print(f"Error generating annotated video: {e}")

def main(yaml_file):
    # Load directories from YAML file
    with open(yaml_file, 'r') as file:
        config = yaml.safe_load(file)
    
    directories = config.get('directories', [])
    
    if not directories:
        print("No directories found in the YAML file.")
        return
    
    start_time = time.time()  # Start timer for the whole process

    for input_dir in directories:
        input_dir = Path(input_dir)
        if not input_dir.exists():
            print(f"Directory does not exist: {input_dir}")
            continue
        
        # Preprocess the videos
        prepare_videos(input_dir, use_gpu=True)

        # Run SLEAP tracking and annotation on preprocessed videos
        #run_sleap_on_preprocessed_videos(input_dir, model_path, make_annotated_video=False)

    # End timer for the entire process
    end_time = time.time()
    print(f"Total time taken for the whole process: {end_time - start_time:.2f} seconds")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python preprocess_and_track.py <config.yaml>")
        sys.exit(1)
    
    yaml_file = sys.argv[1]
    main(yaml_file)