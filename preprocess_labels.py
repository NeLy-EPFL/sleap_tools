import sleap
import cv2
from pathlib import Path

# Template size
template_width = 96
template_height = 516

# Load the existing SLEAP project
labels = sleap.Labels.load_file(
    "/mnt/upramdya_data/MD/MultiMazeRecorder/Videos/231115_TNT_Fine_2_Videos_Tracked/arena9/corridor6/corridor6_tracked_ball.slp"
)

original_video_path = Path("/mnt/upramdya_data/MD/MultiMazeRecorder/Videos/231115_TNT_Fine_2_Videos_Tracked/arena9/corridor6/corridor6.mp4")
original_video = original_video_path.as_posix()

preprocessed_video_path = Path("/mnt/upramdya_data/MD/MultiMazeRecorder/Videos/231115_TNT_Fine_2_Videos_Tracked/arena9/corridor6/corridor6_preprocessed.mp4")

# Define the preprocessing functions
def resize_labels(labels, original_width, original_height, new_width, new_height):
    """Resize the labels according to the new frame size."""
    x_scale = new_width / original_width
    y_scale = new_height / original_height
    return [(int(x * x_scale), int(y * y_scale)) for x, y in labels]


def apply_arena_mask_to_labels(labels, mask_padding, crop_top, crop_bottom, new_height):
    """Adjust the labels according to the cropping and padding applied to the frame."""
    # Crop from top and bottom
    cropped_labels = [
        (x, y - crop_top)
        for x, y in labels
        if crop_top <= y < (new_height - crop_bottom)
    ]

    # Add padding to the left and right
    padded_labels = [(x + mask_padding, y) for x, y in cropped_labels]

    return padded_labels


def preprocess_labels(
    labels,
    original_width,
    original_height,
    template_width,
    template_height,
    mask_padding=20,
    crop_top=74,
    crop_bottom=0,
):
    """Preprocess the labels by resizing and applying mask transformations."""
    # Resize the labels according to the new frame size
    resized_labels = resize_labels(
        labels, original_width, original_height, template_width, template_height
    )

    # Apply arena mask transformations
    final_labels = apply_arena_mask_to_labels(
        resized_labels, mask_padding, crop_top, crop_bottom, template_height
    )

    return final_labels




print(f"Processing video: {original_video_path} -> {preprocessed_video_path}")

# Load the preprocessed video to get its dimensions
cap = cv2.VideoCapture(preprocessed_video_path.as_posix())
ret, frame = cap.read()
preprocessed_height, preprocessed_width, _ = frame.shape
cap.release()

# Get the original video dimensions
cap = cv2.VideoCapture(original_video_path.as_posix())
ret, frame = cap.read()
original_height, original_width, _ = frame.shape
cap.release()

# Add the preprocessed video to the labels project
preprocessed_video = sleap.Video.from_filename(
    preprocessed_video_path.as_posix()
)
labels.videos.append(preprocessed_video)

# Preprocess labels associated with this video
frames = labels.get(original_video)
preprocessed_labeled_frames = []

for labeled_frame in frames:
    src_labels = [
        (p.x, p.y)
        for instance in labeled_frame.instances
        for p in instance.points
    ]

    # Preprocess the labels using the working function
    preprocessed_labels = preprocess_labels(
        src_labels,
        original_width=original_width,
        original_height=original_height,
        template_width=template_width,
        template_height=template_height,
        mask_padding=20,  # Adjust based on your preprocessing settings
        crop_top=74,  # Adjust based on your preprocessing settings
        crop_bottom=0,  # Adjust based on your preprocessing settings
    )

    # Create instances for the preprocessed labels
    new_instances = []
    for instance in labeled_frame.instances:
        for node, (x, y) in zip(instance.points, preprocessed_labels):
            node.x, node.y = x, y
        new_instances.append(instance)

    # Create a new labeled frame with the preprocessed video and new instances
    new_labeled_frame = sleap.LabeledFrame(
        video=preprocessed_video,
        frame_idx=labeled_frame.frame_idx,
        instances=new_instances,
    )

    # Add the new labeled frame to the list
    preprocessed_labeled_frames.append(new_labeled_frame)

# Add the new labeled frames to the project
labels.labeled_frames.extend(preprocessed_labeled_frames)


print(f"Finished processing video: {original_video_path}")

# Remove old videos and labels associated with them

labels.remove_video(original_video_path)

# Save the updated SLEAP project with preprocessed labels
labels.save(
    "/mnt/upramdya_data/_Tracking_models/Sleap/mazerecorder/FlyTracking/FullBody/PreprocessedLabels.slp"
)