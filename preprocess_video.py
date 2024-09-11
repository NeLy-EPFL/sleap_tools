import os
import cv2
import numpy as np
import subprocess
import tempfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from tqdm import tqdm


def resize_frame(frame, width, height):
    """Resize the frame to the given width and height."""
    return cv2.resize(frame, (width, height))


def histogram_stretching(frame):
    """Enhance contrast using histogram stretching."""
    min_val, max_val = np.min(frame), np.max(frame)
    if max_val > min_val:
        return np.clip((frame - min_val) / (max_val - min_val) * 255, 0, 255).astype(
            np.uint8
        )
    return frame


def binarise(frame):
    """Detect the corridors in a frame using a simple threshold."""
    if len(frame.shape) == 3:  # Convert to grayscale if needed
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(frame, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    kernel = np.ones((60, 20), np.uint8)  # Smaller kernel to avoid losing details
    closing = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    return closing


def create_arena_mask(binary_frame):
    """Create a mask that keeps only the area inside the detected arena."""
    contours, _ = cv2.findContours(
        binary_frame, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    mask = np.zeros_like(binary_frame)
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        cv2.drawContours(mask, [largest_contour], -1, (255), thickness=cv2.FILLED)
    return mask


def apply_arena_mask(frame, mask, dilation_iterations=1, padding=True, cropping=True):
    """Apply the arena mask to the frame and optionally crop/pad the result."""
    kernel = np.ones((5, 5), np.uint8)
    dilated_mask = cv2.dilate(mask, kernel, iterations=dilation_iterations)
    masked_frame = cv2.bitwise_and(frame, frame, mask=dilated_mask)

    if cropping:
        cropped_frame = masked_frame[74:, :]
    else:
        cropped_frame = masked_frame

    if padding:
        padded_frame = cv2.copyMakeBorder(
            cropped_frame, 0, 0, 20, 20, cv2.BORDER_CONSTANT, value=[0, 0, 0]
        )
    else:
        padded_frame = cropped_frame

    return padded_frame


def preprocess_frame(frame, mask, width, height):
    """Preprocess the frame by resizing, stretching histograms, and applying mask."""
    resized_frame = resize_frame(frame, width, height)
    stretched_frame = histogram_stretching(resized_frame)
    masked_frame = apply_arena_mask(stretched_frame, mask, dilation_iterations=2)
    final_frame = histogram_stretching(masked_frame)
    return final_frame


def process_video_chunk(chunk, mask, width, height):
    """Process a chunk of frames."""
    return [preprocess_frame(frame, mask, width, height) for frame in chunk]


def process_video(
    input_path,
    output_path,
    template_width,
    template_height,
    sample_frames=None,
    chunk_size=100,
):
    # Check if the output video already exists
    if output_path.exists():
        print(f"Preprocessed video already exists: {output_path}")
        return

    try:
        output_dir = output_path.parent
        output_dir.mkdir(parents=True, exist_ok=True)

        cap = cv2.VideoCapture(str(input_path))
        if not cap.isOpened():
            print(f"Error: Could not open video {input_path}.")
            return

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if sample_frames:
            total_frames = min(total_frames, sample_frames)

        # Read the last frame to create the mask
        cap.set(cv2.CAP_PROP_POS_FRAMES, total_frames - 1)
        ret, last_frame = cap.read()
        if not ret:
            print("Error: Could not read the last frame.")
            return

        resized_last_frame = resize_frame(last_frame, template_width, template_height)
        binary_last_frame = binarise(resized_last_frame)
        arena_mask = create_arena_mask(binary_last_frame)

        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Reset the frame position

        # Use a temporary directory for storing frames
        temp_dir = tempfile.mkdtemp()

        chunk_idx = 0
        frames_to_process = []
        frames_processed = 0  # Counter for processed frames

        with ProcessPoolExecutor() as executor:
            futures = []
            with tqdm(
                total=total_frames, desc="Processing frames", unit="frame"
            ) as pbar:
                while True:
                    ret, frame = cap.read()
                    if not ret or (sample_frames and frames_processed >= sample_frames):
                        break
                    frames_to_process.append(frame)

                    if len(frames_to_process) == chunk_size:
                        futures.append(
                            executor.submit(
                                process_video_chunk,
                                frames_to_process,
                                arena_mask,
                                template_width,
                                template_height,
                            )
                        )
                        frames_to_process = []

                    # Process completed futures
                    for future in as_completed(futures):
                        processed_frames = future.result()
                        for preprocessed_frame in processed_frames:
                            frame_filename = os.path.join(
                                temp_dir, f"frame_{chunk_idx:06d}.png"
                            )
                            cv2.imwrite(frame_filename, preprocessed_frame)
                            chunk_idx += 1
                        pbar.update(len(processed_frames))
                        frames_processed += len(processed_frames)

                    futures = []

                # Process any remaining frames
                if frames_to_process:
                    futures.append(
                        executor.submit(
                            process_video_chunk,
                            frames_to_process,
                            arena_mask,
                            template_width,
                            template_height,
                        )
                    )

                for future in as_completed(futures):
                    processed_frames = future.result()
                    for preprocessed_frame in processed_frames:
                        frame_filename = os.path.join(
                            temp_dir, f"frame_{chunk_idx:06d}.png"
                        )
                        cv2.imwrite(frame_filename, preprocessed_frame)
                        chunk_idx += 1
                    pbar.update(len(processed_frames))

        # Assemble video using ffmpeg
        ffmpeg_command = (
            f"ffmpeg -loglevel panic -nostats -hwaccel cuda -framerate 29 -i {os.path.join(temp_dir, 'frame_%06d.png')} "
            f"-vsync 0 -frames:v {frames_processed} -pix_fmt yuv420p -c:v libx265 -crf 15 {output_path.as_posix()}"
        )
        ffmpeg_result = subprocess.call(ffmpeg_command, shell=True)

        if ffmpeg_result == 0:
            print(f"Video assembled successfully: {output_path}")
        else:
            print("Error assembling video.")

    finally:
        # Clean up temporary files
        cap.release()
        if "temp_dir" in locals():
            for file in os.listdir(temp_dir):
                os.remove(os.path.join(temp_dir, file))
            os.rmdir(temp_dir)


def process_all_videos(
    input_dir, output_dir, template_width, template_height, sample_frames=None
):
    """Process all videos in the input directory and its subdirectories."""
    for input_path in Path(input_dir).rglob("*.mp4"):
        # Skip files that are already preprocessed
        if "_preprocessed" in input_path.stem:
            print(f"Skipping preprocessed video: {input_path}")
            continue

        relative_path = input_path.relative_to(input_dir)
        output_path = Path(output_dir) / relative_path
        output_path = output_path.with_name(output_path.stem + "_preprocessed.mp4")

        print(f"Processing video: {input_path}")
        process_video(
            input_path, output_path, template_width, template_height, sample_frames
        )
        print(f"Completed video: {output_path}")


def process_videos_in_directory(
    input_directory,
    output_directory,
    template_width=96,
    template_height=516,
    sample_frames=None,
):
    """Process all videos in the specified directory."""
    input_directory = Path(input_directory)
    output_directory = Path(output_directory)

    process_all_videos(
        input_directory,
        output_directory,
        template_width,
        template_height,
        sample_frames,
    )

    print(f"Preprocessing complete for all videos in: {input_directory}")


# Template size
template_width = 96
template_height = 516

# Input and output directories
input_directory = Path("/path/to/input_directory")
output_directory = Path("/path/to/output_directory")

# Number of frames to process for the sample video (set to None to process the entire video)
sample_frames = None

# Process all videos in the input directory
process_videos_in_directory(
    input_directory, output_directory, template_width, template_height, sample_frames
)

print(f"Preprocessing complete for all videos in: {input_directory}")
