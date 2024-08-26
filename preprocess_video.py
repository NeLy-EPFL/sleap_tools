import cv2
import numpy as np
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm


def resize_frame(frame, width, height):
    """Resize the frame to the given width and height."""
    return cv2.resize(frame, (width, height))


def equalize_histogram(frame):
    """Apply adaptive histogram equalization to the frame."""
    if len(frame.shape) == 3:  # Convert to grayscale if needed
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(frame)


def binarise(frame):
    """Detect the corridors in a frame using a simple threshold."""
    if len(frame.shape) == 3:  # Convert to grayscale if needed
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(frame, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Add some erosion and dilation to remove noise
    kernel = np.ones((50, 10), np.uint8)
    closing = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    return closing


def create_arena_mask(binary_frame):
    """Create a mask that keeps only the area inside the detected arena."""
    # Find contours in the binary frame
    contours, _ = cv2.findContours(
        binary_frame, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    # Create an empty mask
    mask = np.zeros_like(binary_frame)

    # Assume the largest contour is the arena
    largest_contour = max(contours, key=cv2.contourArea)

    # Draw the largest contour on the mask
    cv2.drawContours(mask, [largest_contour], -1, (255), thickness=cv2.FILLED)

    return mask


def apply_arena_mask(frame, mask):
    """Apply the arena mask to the frame and crop/pad the result."""
    # Apply the mask to the frame
    masked_frame = cv2.bitwise_and(frame, frame, mask=mask)

    # Crop 10 pixels from top and bottom
    cropped_frame = masked_frame[6:-6, :]

    # Add 10 pixels of black padding to the left and right
    padded_frame = cv2.copyMakeBorder(
        cropped_frame, 0, 0, 20, 20, cv2.BORDER_CONSTANT, value=[0, 0, 0]
    )

    return padded_frame


def preprocess_frame(frame, mask, width, height):
    """Preprocess the frame by resizing, equalizing histogram, and applying mask."""
    resized_frame = resize_frame(frame, width, height)
    equalized_frame = equalize_histogram(resized_frame)
    final_frame = apply_arena_mask(equalized_frame, mask)  # Fixed: should use equalized_frame
    return final_frame


def process_frame(frame, mask, width, height):
    """Wrapper function to preprocess a single frame."""
    return preprocess_frame(frame, mask, width, height)


def process_video(input_path, output_dir, template_width, template_height):
    """Process the entire video and save the preprocessed frames."""
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {input_path}.")
        return

    try:
        # Get video properties
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Read the last frame to generate the mask
        cap.set(cv2.CAP_PROP_POS_FRAMES, total_frames - 1)
        ret, last_frame = cap.read()
        if not ret:
            print("Error: Could not read the last frame.")
            return

        # Preprocess the last frame to generate the mask
        resized_last_frame = resize_frame(last_frame, template_width, template_height)
        equalized_last_frame = equalize_histogram(resized_last_frame)
        binary_last_frame = binarise(equalized_last_frame)
        arena_mask = create_arena_mask(binary_last_frame)

        # Reset the video capture to the first frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)

        # Process frames with multithreading and progress bar
        with ThreadPoolExecutor() as executor:
            futures = []
            for frame_idx in range(total_frames):
                ret, frame = cap.read()
                if not ret:
                    print(f"Error: Could not read frame {frame_idx}.")
                    break
                future = executor.submit(
                    process_frame,
                    frame,
                    arena_mask,
                    template_width,
                    template_height,
                )
                futures.append((frame_idx, future))

            for frame_idx, future in tqdm(
                futures, total=total_frames, desc="Processing frames"
            ):
                preprocessed_frame = future.result()
                if preprocessed_frame is None:
                    print(f"Warning: Preprocessed frame {frame_idx} is None.")
                    continue
                frame_filename = os.path.join(output_dir, f"frame_{frame_idx:06d}.png")
                cv2.imwrite(frame_filename, preprocessed_frame)

        print(f"Preprocessed frames saved to: {output_dir}")

    finally:
        # Release resources
        cap.release()


# Template size
template_width = 96
template_height = 516

# Input video path and output directory
input_video_path = "/mnt/upramdya_data/MD/MultiMazeRecorder/Videos/231129_TNT_Fine_3_Videos_Tracked/arena4/corridor2/corridor2.mp4"
output_frames_dir = "/mnt/upramdya_data/MD/MultiMazeRecorder/Videos/231129_TNT_Fine_3_Videos_Tracked/arena4/corridor2/frames"

# Process the video
process_video(input_video_path, output_frames_dir, template_width, template_height)

print(f"Preprocessing complete for video: {input_video_path}")

# Assemble the frames into a video with a ffmpeg command
ffmpeg_command = (
    f"ffmpeg -y -r 29 -pattern_type glob -i '{output_frames_dir}/*.png' "
    f"-c:v libx264 -vf fps=29 -pix_fmt yuv420p {output_frames_dir}/output.mp4"
)
ffmpeg_result = os.system(ffmpeg_command)

# Check if FFmpeg command was successful before removing frames
if ffmpeg_result == 0:
    print("Video assembled successfully.")
    os.system(f"rm -r {output_frames_dir}")
else:
    print("Error assembling video. Frames not removed.")
