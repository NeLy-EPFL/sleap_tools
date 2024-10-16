import os
import cv2
import numpy as np
import subprocess
import tempfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from tqdm import tqdm
import queue
import time

import threading

from matplotlib import pyplot as plt

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


def create_arena_mask(binary_frame, dilation_iterations=1):
    """Create a mask that keeps only the area inside the detected arena."""
    contours, _ = cv2.findContours(
        binary_frame, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    mask = np.zeros_like(binary_frame)
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        cv2.drawContours(mask, [largest_contour], -1, (255), thickness=cv2.FILLED)
        
    kernel = np.ones((5, 5), np.uint8)
    dilated_mask = cv2.dilate(mask, kernel, iterations=dilation_iterations)
    
    return dilated_mask


def apply_arena_mask(frame, masked_frame, padding=True, cropping=True):
    """Apply the arena mask to the frame and optionally crop/pad the result."""
    
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


def preprocess_frame(frame, mask, width, height, isolated=False):
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
                            if preprocessed_frame is not None:
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
                        if preprocessed_frame is not None:
                            frame_filename = os.path.join(
                                temp_dir, f"frame_{chunk_idx:06d}.png"
                            )
                            cv2.imwrite(frame_filename, preprocessed_frame)
                            chunk_idx += 1
                    pbar.update(len(processed_frames))

        # Assemble video using ffmpeg
        ffmpeg_command = (
            f"ffmpeg -loglevel error -nostats -hwaccel cuda -framerate 29 -i {os.path.join(temp_dir, 'frame_%06d.png')} "
            f"-vsync 0 -frames:v {frames_processed} -pix_fmt yuv420p -c:v libx265 -crf 15 {output_path.as_posix()}"
        )
        print(f"Running ffmpeg command: {ffmpeg_command}")
        ffmpeg_result = subprocess.run(
            ffmpeg_command, shell=True, capture_output=True, text=True
        )

        if ffmpeg_result.returncode == 0:
            print(f"Video assembled successfully: {output_path}")
        else:
            print("Error assembling video.")
            print("ffmpeg output:", ffmpeg_result.stdout)
            print("ffmpeg error:", ffmpeg_result.stderr)

    finally:
        # Clean up temporary files
        cap.release()
        if "temp_dir" in locals():
            for file in os.listdir(temp_dir):
                os.remove(os.path.join(temp_dir, file))
            os.rmdir(temp_dir)


### GPU version ###
    
def resize_frame_gpu(frame_gpu, width, height):
    # Ensure that frame_gpu is of type cv2.cuda.GpuMat
    if not isinstance(frame_gpu, cv2.cuda.GpuMat):
        print(f"Error: frame_gpu is not a cv2.cuda.GpuMat, got {type(frame_gpu)}.")
        return None

    # Check if frame_gpu is empty
    if frame_gpu.empty():
        print("Error: frame_gpu is empty.")
        return None

    try:
        # Create a new GpuMat for the resized frame
        resized_frame_gpu = cv2.cuda_GpuMat()

        # Perform the resize operation
        resized_frame_gpu = cv2.cuda.resize(frame_gpu, (width, height))

        return resized_frame_gpu  # Return the newly created GpuMat
    except Exception as e:
        print(f"Resize failed: {e}")
        return None

def histogram_stretching_gpu(gpu_image):
    # Convert to grayscale if necessary
    if gpu_image.channels() == 3:
        gray_gpu_frame = cv2.cuda.cvtColor(gpu_image, cv2.COLOR_BGR2GRAY)
    else:
        gray_gpu_frame = gpu_image

    if gray_gpu_frame.empty():
        print("Error: gray_gpu_frame is empty after conversion.")
        return None

    # Create a new GpuMat for the stretched result
    stretched_image = cv2.cuda_GpuMat(gray_gpu_frame.size(), gray_gpu_frame.type())

    # Perform histogram stretching
    cv2.cuda.normalize(src = gray_gpu_frame, dst = stretched_image, alpha= 0, beta = 255, norm_type = cv2.NORM_MINMAX, dtype = -1)

    return stretched_image  # Return the newly created GpuMat

def apply_arena_mask_gpu(frame_gpu, mask, padding=True, cropping=True):
    """Apply the arena mask to the frame and optionally crop/pad the result using GPU."""
    
    # Upload the mask to the GPU
    mask_gpu = cv2.cuda_GpuMat()
    mask_gpu.upload(mask)
    
    # Ensure the mask is of the correct type
    if mask_gpu.type() != cv2.CV_8UC1:
        raise ValueError("Mask must be a single-channel 8-bit image (CV_8UC1).")
    
    if frame_gpu.type() != cv2.CV_8UC1:
        raise ValueError("Frame must be a single-channel 8-bit image (CV_8UC1).")

    # Ensure the mask size matches the frame size
    if mask_gpu.size() != frame_gpu.size():
        raise ValueError("Mask size must match the frame size.")
    
    # Create a new GpuMat for the masked frame (fresh allocation)
    masked_frame_gpu = cv2.cuda_GpuMat(frame_gpu.size(), cv2.CV_8UC1)
    
    # Apply the mask to the frame on the GPU
    cv2.cuda.bitwise_and(src1=frame_gpu, src2=frame_gpu, dst=masked_frame_gpu, mask=mask_gpu)

    # Cropping logic
    size = masked_frame_gpu.size()
    rows, cols = size[1], size[0]
    
    minY = 74
    maxY = rows
    minX = 0
    maxX = cols

    if cropping:
        # Crop using a new GpuMat
        cropped_frame_gpu = masked_frame_gpu.rowRange(minY, maxY).colRange(minX, maxX)
    else:
        cropped_frame_gpu = masked_frame_gpu

    # Padding logic
    if padding:
        # Create a new GpuMat for the padded frame
        # Ensure padding areas are set to [0, 0, 0] (black) explicitly
        padded_frame_gpu = cv2.cuda.copyMakeBorder(
            cropped_frame_gpu, top=0, bottom=0, left=20, right=20, borderType=cv2.BORDER_CONSTANT, value=[0]
        )
    else:
        padded_frame_gpu = cropped_frame_gpu

    # Release unused GPU memory
    masked_frame_gpu.release()
    cropped_frame_gpu.release()

    return padded_frame_gpu  # Return the newly created GpuMat


def preprocess_frame_gpu(frame_gpu, mask, width, height, isolated=False):
    """Preprocess the frame by resizing, stretching histograms, and applying the provided mask."""

    # Resize the frame on the GPU
    resized_frame_gpu = resize_frame_gpu(frame_gpu, width, height)
    if resized_frame_gpu is None:
        print("Error: Resized frame is None, skipping frame.")
        return None

    # Stretch the histogram of the resized frame
    stretched_frame_gpu = histogram_stretching_gpu(resized_frame_gpu)
    if stretched_frame_gpu is None:
        print("Error: Stretched frame is None, skipping frame.")
        return None

    # Apply the arena mask to the stretched frame
    preprocessed_frame_gpu = apply_arena_mask_gpu(stretched_frame_gpu, mask)
    if preprocessed_frame_gpu is None:
        print("Error: Preprocessed frame is None, skipping frame.")
        return None

    return preprocessed_frame_gpu  # Return the final preprocessed GpuMat

def process_video_gpu_CV(input_path, output_path, template_width, template_height, sample_frames=None, display_interval=100):
    """Process video using GPU with mask generation."""
    
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

        # Generate the arena mask
        resized_last_frame = resize_frame(last_frame, template_width, template_height)
        binary_last_frame = binarise(resized_last_frame)
        arena_mask = create_arena_mask(binary_last_frame)
        
        # # Upload the mask to the GPU ONCE
        # arena_mask_gpu = cv2.cuda_GpuMat()
        # arena_mask_gpu.upload(arena_mask)

        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Reset the frame position

        # Prepare to write to output video
        fourcc = cv2.VideoWriter_fourcc(*"H264")
        fps = cap.get(cv2.CAP_PROP_FPS)
        out = cv2.VideoWriter(str(output_path), fourcc, fps, (136, 442), isColor=False)

        if not out.isOpened():
            print(f"Error: Could not open video writer for {output_path}.")
            return

        # Process frames
        frame_count = 0
        with tqdm(total=total_frames, desc="Processing frames", unit="frame") as pbar:
            while cap.isOpened():
                ret, img = cap.read()
                if not ret:
                    print("No more frames to read or error in reading.")
                    break
                
                # Ensure the frame is grayscale
                if img.ndim == 3:
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

                # Upload frame to GPU
                gpu_img = cv2.cuda_GpuMat()
                gpu_img.upload(img)

                # Process the frame
                preprocessed_frame_gpu = preprocess_frame_gpu(gpu_img, arena_mask, template_width, template_height)
                if preprocessed_frame_gpu is not None:
                    output_frame = preprocessed_frame_gpu.download()  # Download processed frame from GPU
                    
                    # Ensure output frame size is correct
                    if output_frame.shape != (442, 136):
                        print(f"Warning: Frame size mismatch. Expected {(136, 442)}, got {output_frame.shape}.")
                    if output_frame.dtype != np.uint8:
                        print(f"Warning: Frame type mismatch. Expected np.uint8, got {output_frame.dtype}.")

                    out.write(output_frame)  # Write to video only processed frames
                else:
                    print(f"Warning: Preprocessed frame is None, skipping frame {frame_count}.")
                    
                # reset the GPU memory
                # After each frame processing, release GpuMat memory
                gpu_img.release()
                preprocessed_frame_gpu.release()

                frame_count += 1
                pbar.update(1)

        cap.release()
        out.release()
        cv2.destroyAllWindows()  # Close all OpenCV windows
        print(f"Preprocessed video saved to: {output_path}")

    except Exception as e:
        print(f"Error during processing: {e}")

def process_video_gpu_ffmpeg(input_path, output_path, template_width, template_height, sample_frames=None):
    """Process video using GPU and pipe frames directly to FFmpeg with CRF 15 for high-quality output."""
    
    output_path = Path(output_path)
    if output_path.exists():
        print(f"Preprocessed video already exists: {output_path}")
        return

    try:
        cap = cv2.VideoCapture(str(input_path))
        if not cap.isOpened():
            print(f"Error: Could not open video {input_path}.")
            return

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if sample_frames:
            total_frames = min(total_frames, sample_frames)

        cap.set(cv2.CAP_PROP_POS_FRAMES, total_frames - 1)
        ret, last_frame = cap.read()
        if not ret:
            print("Error: Could not read the last frame.")
            return

        resized_last_frame = resize_frame(last_frame, template_width, template_height)
        binary_last_frame = binarise(resized_last_frame)
        arena_mask = create_arena_mask(binary_last_frame)

        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

        ffmpeg_command = [
            "ffmpeg", "-y", "-loglevel", "error", "-nostats", "-hwaccel", "cuda",
            "-f", "rawvideo", "-pixel_format", "gray", 
            "-video_size", f"{136}x{442}",
            "-framerate", "29", "-i", "pipe:", "-c:v", "libx265", 
            "-crf", "15", "-pix_fmt", "yuv420p", output_path.as_posix(),
            "-report"
        ]

        # Start the FFmpeg process
        process = subprocess.Popen(ffmpeg_command, stdin=subprocess.PIPE, stderr=subprocess.PIPE)

        frame_count = 0
        with tqdm(total=total_frames, desc="Processing frames", unit="frame") as pbar:
            while cap.isOpened():
                ret, img = cap.read()
                if not ret:
                    print(f"No more frames to read or error in reading at frame count: {frame_count}")
                    break
                
                img = convert_to_grayscale(img)
                if img is None:
                    continue

                gpu_img = cv2.cuda_GpuMat()
                gpu_img.upload(img)

                preprocessed_frame_gpu = preprocess_frame_gpu(gpu_img, arena_mask, template_width, template_height)
                if preprocessed_frame_gpu is not None:
                    output_frame = preprocessed_frame_gpu.download()
                    
                    if output_frame.ndim != 2:  # Ensure it's grayscale
                        print(f"Warning: Expected 2D array for grayscale but got shape {output_frame.shape}. Skipping frame.")
                        continue
                    
                    try:
                        process.stdin.write(output_frame.tobytes())
                    except BrokenPipeError:
                        print("Broken pipe: FFmpeg process has likely terminated.")
                        break
                else:
                    print(f"Warning: Preprocessed frame is None, skipping frame {frame_count}.")
                
                gpu_img.release()
                preprocessed_frame_gpu.release()

                frame_count += 1
                pbar.update(1)

        cap.release()
        process.stdin.close()  
        stdout, stderr = process.communicate()  

        if process.returncode != 0:
            print("FFmpeg error:", stderr.decode())  # Decode stderr for human-readable output
        else:
            print(f"Preprocessed video saved to: {output_path}")

    except Exception as e:
        print(f"Error during processing: {e}")
    
    finally:
        cap.release()

def convert_to_grayscale(image):
    """Convert an image to grayscale if it's not already."""
    if image.ndim == 3:  # Check if the image is not grayscale
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return image

def process_all_videos(
    input_dir,
    output_dir,
    template_width,
    template_height,
    sample_frames=None,
    exclude_keywords=None,
    use_gpu=False,
):
    """Process all videos in the input directory and its subdirectories using CPU or GPU."""
    if exclude_keywords is None:
        exclude_keywords = []

    for input_path in Path(input_dir).rglob("*.mp4"):
        # Skip files that are already preprocessed or contain exclude keywords
        if any(keyword in input_path.stem for keyword in exclude_keywords):
            print(f"Skipping video due to exclude keyword: {input_path}")
            continue

        # Determine the output path
        if output_dir:
            relative_path = input_path.relative_to(input_dir)
            output_path = Path(output_dir) / relative_path
            output_path = output_path.with_name(output_path.stem + "_preprocessed.mp4")
        else:
            output_path = input_path.with_name(input_path.stem + "_preprocessed.mp4")
            
        # If the output file already exists, skip processing
        if output_path.exists():
            print(f"Preprocessed video already exists: {output_path}")
            continue

        print(f"Processing video: {input_path}")

        if use_gpu:
            # process_video_gpu_CV( # Use the GPU version
            #     input_path, output_path, template_width, template_height, sample_frames
            # )
            process_video_gpu_ffmpeg(
                input_path, output_path, template_width, template_height, sample_frames
            )
        else:
            process_video(
                input_path, output_path, template_width, template_height, sample_frames
            )

        print(f"Completed video: {output_path}")


def process_videos_in_directory(
    input_directory,
    output_directory=None,
    template_width=96,
    template_height=516,
    sample_frames=None,
    exclude_keywords=None,
    use_gpu=False,
):
    """Process all videos in the specified directory using CPU or GPU."""
    input_directory = Path(input_directory)
    output_directory = Path(output_directory) if output_directory else None

    process_all_videos(
        input_directory,
        output_directory,
        template_width,
        template_height,
        sample_frames,
        exclude_keywords,
        use_gpu,  # Pass the GPU flag
    )

    print(f"Preprocessing complete for all videos in: {input_directory}")


# # Template size
# template_width = 96
# template_height = 516

# # Input and output directories
# input_directory = Path(
#     "/home/durrieu/Videos/corridor4"
# )
# output_directory = Path(
#     "/home/durrieu/Videos/corridor4"
# )

# Number of frames to process for the sample video (set to None to process the entire video)
#sample_frames = None

# # Process all videos in the input directory
# process_videos_in_directory(
#     input_directory,
#     output_directory,
#     template_width,
#     template_height,
#     sample_frames,
#     use_gpu=True,
# )

#print(f"Preprocessing complete for all videos in: {input_directory}")


# def reverse_resize_labels(
#     labels, original_width, original_height, template_width, template_height
# ):
#     """Reverse the resizing of labels to the original frame size."""
#     scale_x = original_width / template_width
#     scale_y = original_height / template_height
#     reversed_labels = labels.copy()
#     reversed_labels[:, :, 0] *= scale_x
#     reversed_labels[:, :, 1] *= scale_y
#     return reversed_labels


# def reverse_apply_arena_mask_to_labels(
#     labels, mask_padding=20, crop_top=74, crop_bottom=0, template_height=516
# ):
#     """Reverse the arena mask transformations on the labels."""
#     reversed_labels = labels.copy()
#     reversed_labels[:, :, 1] += crop_top  # Reverse the cropping
#     return reversed_labels


# def reverse_preprocess_labels(
#     labels,
#     original_width,
#     original_height,
#     template_width,
#     template_height,
#     mask_padding=20,
#     crop_top=74,
#     crop_bottom=0,
# ):
#     """Reverse the preprocessing of labels by reversing resizing and mask transformations."""
#     # Reverse the arena mask transformations
#     reversed_labels = reverse_apply_arena_mask_to_labels(
#         labels, mask_padding, crop_top, crop_bottom, template_height
#     )

#     # Reverse the resizing of labels to the original frame size
#     final_labels = reverse_resize_labels(
#         reversed_labels,
#         original_width,
#         original_height,
#         template_width,
#         template_height,
#     )

#     return final_labels
