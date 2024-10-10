from event_detection import detect_events
import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import tkinter as tk
from tkinter import filedialog

event_names = {
    0: 'Address',
    1: 'Toe-up',
    2: 'Mid-backswing',
    3: 'Top',
    4: 'Mid-downswing',
    5: 'Impact',
    6: 'Mid-follow-through',
    7: 'Finish'
}

landmark_names_dict = {0 : 'Head',
11 : 'Left shoulder',
12 : 'Right shoulder',
13 : 'Left elbow',
14 : 'Right elbow',
15 : 'Left wrist',
16 : 'Right wrist',
23 : 'Left hip',
24 : 'Right hip',
25 : 'Left knee',
26 : 'Right knee',
27 : 'Left ankle',
28 : 'Right ankle'
}

def get_events(user_video_path, swing_view, reference_video_path=None):
    """
    Extract events from the user video and a reference video.

    Args:
    - user_video_path (str): Path to the user's video.
    - swing_view (str): Swing view type ('down_the_line' or 'face_on').
    - reference_video_path (str, optional): Path to the reference video.

    Returns:
    - Tuple[List[str], List[List[Event]]]: Paths of the reference and user videos, and the detected events.
    """
    if reference_video_path is None:
            if swing_view == 'down_the_line':
                reference_video_path = "rory_mcilroy_down_the_line.mp4"
            elif swing_view == 'face_on':
                reference_video_path = "rory_mcilroy_face_on.mp4"

    user_events = detect_events(user_video_path)
    reference_events = detect_events(reference_video_path)

    video_paths = [reference_video_path, user_video_path]
    events = [reference_events, user_events]

    return video_paths, events

def videos_to_pose(video_paths, events, toggle_landmarks=False, save_image=False):
    """
    Extract pose landmarks from specified frames in multiple videos and save to CSV files.

    Args:
        video_paths (list): List of video file paths.
        events (list): List of lists, where each sublist contains frame indices for the corresponding video.

    Returns:
        Landmarks and image
    """
    # Initialize MediaPipe Pose
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose()

    def extract_landmarks(results):
        """Extract pose landmarks from MediaPipe results."""
        if results.pose_landmarks:
            landmarks = []
            for landmark in results.pose_landmarks.landmark:
                landmarks.append((landmark.x, landmark.y))
            return landmarks
        return None

    def landmarks_to_dict(landmarks):
        """Convert list of landmarks to dictionary format for DataFrame."""
        if landmarks:
            return {f'landmark_{i}_{axis}': coord for i, (x, y) in enumerate(landmarks) for axis, coord in zip(['x', 'y'], [x, y])}
        return {}

    def resize_image(image, target_width=160):
        """
        Resize the input image to a specified width while maintaining the aspect ratio.
        """
        height, width, _ = image.shape
        aspect_ratio = height / width
        new_height = int(target_width * aspect_ratio)
        resized_image = cv2.resize(image, (target_width, new_height))
        return resized_image

    video_landmarks = {}
    composites = []
    for video_path, frames_to_extract in zip(video_paths, events):
        cap = cv2.VideoCapture(video_path)
        landmarks_list = []
        imgs = []

        for idx, frame_idx in enumerate(frames_to_extract):
            # Set frame position
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)

            ret, frame = cap.read()
            if not ret:
                print(f"Failed to read frame at index {frame_idx} in video {video_path}")
                continue

            # Convert frame to RGB
            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Process the image
            results = pose.process(image_rgb)

            # Extract pose landmarks
            landmarks = extract_landmarks(results)

            # Convert landmarks to a dictionary and append to list
            landmarks_dict = landmarks_to_dict(landmarks)
            landmarks_dict['frame_idx'] = frame_idx
            landmarks_list.append(landmarks_dict)

            if toggle_landmarks == True:
                # Draw landmarks on the frame
                if results.pose_landmarks:
                    mp.solutions.drawing_utils.draw_landmarks(
                        frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,

                    )

            resized_frame = resize_image(frame)

            cv2.putText(resized_frame, event_names[idx], (10, 15), cv2.FONT_HERSHEY_DUPLEX, 0.45, (255, 0, 0))

            imgs.append(resized_frame)


        cap.release()

        composite_image = np.hstack(imgs)
        composites.append(composite_image)

        # Convert list of dictionaries to DataFrame
        landmarks_df = pd.DataFrame(landmarks_list)

        desired_landmarks = [0, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28]

        columns_to_keep = [f'landmark_{i}_{axis}' for i in desired_landmarks for axis in ['x', 'y', 'z']]
        filtered_landmarks_df = landmarks_df.filter(items=columns_to_keep)

        video_landmarks[video_path] = filtered_landmarks_df

    img1 = composites[0]
    img2 = composites[1]

    # Stack resized images vertically
    stacked_img = np.vstack((img1, img2))

    if save_image == True:
        #save matrix/array as image file
        cv2.imwrite(f'{video_paths[1]}_comparison.png', stacked_img)

    return video_landmarks, stacked_img

def normalize_df(df):
    """
    Normalize the DataFrame for golfer height and center hip coordinates.

    Args:
    - df (pd.DataFrame): DataFrame containing landmark coordinates with columns ending in '_x' or '_y'.

    Returns:
    - pd.DataFrame: A normalized copy of the input DataFrame where x and y coordinates are adjusted.
    """
    df_normalized = df.copy()

    scale_factor = df.filter(like='_y').iloc[0].max() - df.filter(like='_y').iloc[0].min()

    #center  to 0,0 for first frame
    x_ref = (df_normalized['landmark_23_x'].values[0] + df_normalized['landmark_24_x'].values[0])/2
    y_ref = (df_normalized['landmark_23_y'].values[0] + df_normalized['landmark_24_y'].values[0])/2

    for col in df_normalized.columns:
        if '_x' in col:
            df_normalized[col] = (df[col] - x_ref) / scale_factor
        elif '_y' in col:
            df_normalized[col] = (df[col] - y_ref) / scale_factor

    return df_normalized

def normalize_dfs(video_landmarks, video_paths):
    """
    Normalize landmark coordinates for a reference and a test video.

    Args:
    - video_landmarks (dict): Dictionary with video paths as keys and landmark DataFrames as values.
    - video_paths (list of str): List containing paths to the reference and test videos.

    Returns:
    - tuple of (pd.DataFrame, pd.DataFrame): Normalized DataFrames for the reference and test videos.
    """
    ref = video_landmarks[video_paths[0]]
    test = video_landmarks[video_paths[1]]
    ref_normalized = normalize_df(ref)
    test_normalized = normalize_df(test)
    return ref_normalized, test_normalized

def euclidean_distances(row, landmarks):
    """
    Compute Euclidean distances of specified landmarks from the origin.

    Args:
    - row (pd.Series): Series containing landmark coordinates.
    - landmarks (list of int): List of landmark indices to calculate distances for.

    Returns:
    - dict: Dictionary with landmark indices as keys and Euclidean distances as values.
    """
    x_coords = [row[f'landmark_{i}_x'] for i in landmarks]
    y_coords = [row[f'landmark_{i}_y'] for i in landmarks]
    distances = np.sqrt(np.array(x_coords)**2 + np.array(y_coords)**2)
    return dict(zip(landmarks, distances))

def calc_distances(ref_normalized, user_normalized):
    """
    Calculate the Euclidean distances between corresponding landmarks of reference and user data.

    Args:
    - ref_normalized (pd.DataFrame): Normalized DataFrame of reference landmarks.
    - user_normalized (pd.DataFrame): Normalized DataFrame of user landmarks.

    Returns:
    - pd.DataFrame: DataFrame of Euclidean distances between reference and user landmarks.
    """
    difference = ref_normalized - user_normalized
    landmarks = [0, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28]
    all_distances = difference.apply(euclidean_distances, landmarks=landmarks, axis=1)
    distances_df = pd.DataFrame(list(all_distances))
    return distances_df

def recommendations(distances_df, n_recommendations=3, swing_event='Full swing'):
    """
    Generate recommendations based on landmark distances and specified swing events.

    Args:
    - distances_df (pd.DataFrame): DataFrame containing distances between reference and user landmarks.
    - n_recommendations (int, optional): Number of top recommendations to return. Default is 3.
    - swing_event (str, optional): Specific swing event to filter recommendations by. Default is 'Full swing'.

    Returns:
    - tuple of (str, str, str): 
        - `at_specified_event`: Recommendations for the specified swing event.
        - `events_differences`: Events with the largest overall differences.
        - `body_differences`: Landmarks with the largest overall differences.
    """
    flattened = distances_df.stack()
    at_specified_event = ""
    events_differences = ""
    body_differences = ""
    
    if swing_event != 'Full swing':
        event = next((k for k, v in event_names.items() if v == swing_event), None)
        flattened = flattened[event]
        top_n = flattened.nlargest(n_recommendations)
        for col, _ in top_n.items():
            at_specified_event += f"{landmark_names_dict[col]} at {event_names[event]}\n"

    else:
        top_n = flattened.nlargest(n_recommendations)
        for (row, col), _ in top_n.items():
            at_specified_event += f"    {landmark_names_dict[col]} at {event_names[row]}\n"

    dissimilar_events = distances_df.sum(axis=1).nlargest(3)
    for row, _ in  dissimilar_events.items():
        events_differences += f"    {event_names[row]}\n"

    dissimilar_landmarks = distances_df.sum(axis=0).nlargest(3)
    for row, _ in  dissimilar_landmarks.items():
        body_differences += f"    {landmark_names_dict[row]}\n"
    return at_specified_event, events_differences, body_differences

def browse_user_video():
    filename = filedialog.askopenfilename()
    user_video_path_var.set(filename)

def browse_reference_video():
    filename = filedialog.askopenfilename()
    reference_video_path_var.set(filename)

def run_analysis():
    user_video_path = user_video_path_var.get()
    swing_view = swing_view_var.get()
    reference_video_path = reference_video_path_var.get() or None
    toggle_landmarks = toggle_landmarks_var.get()
    save_image = save_image_var.get()
    swing_event = swing_event_var.get()
    n_recommendations = int(n_recommendations_var.get())

    video_paths, events = get_events(user_video_path=user_video_path, swing_view=swing_view, reference_video_path=reference_video_path)
    video_landmarks, stacked_img = videos_to_pose(video_paths, events, toggle_landmarks=toggle_landmarks, save_image=save_image)
    ref_normalized, test_normalized = normalize_dfs(video_landmarks, video_paths)
    distances_df = calc_distances(ref_normalized, test_normalized)

    # Capture the recommendations text from the function
    recommendations_a, recommendations_b, recommendations_c = recommendations(distances_df, n_recommendations=n_recommendations, swing_event=swing_event)

    # Display each recommendation in its own Text widget
    recommendations_output_1.delete(1.0, tk.END)
    recommendations_output_1.insert(tk.END, recommendations_a)

    recommendations_output_2.delete(1.0, tk.END)
    recommendations_output_2.insert(tk.END, recommendations_b)

    recommendations_output_3.delete(1.0, tk.END)
    recommendations_output_3.insert(tk.END, recommendations_c)

    # Ensure the GUI updates
    root.update_idletasks()

    # Display the image with OpenCV
    cv2.imshow('Image', stacked_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# Setting up the main application window
root = tk.Tk()
root.title("Swing Analysis Tool")

# GUI Elements
tk.Label(root, text="User Video Path:").grid(row=0, column=0)
user_video_path_var = tk.StringVar()
tk.Entry(root, textvariable=user_video_path_var).grid(row=0, column=1)
tk.Button(root, text="Browse", command=browse_user_video).grid(row=0, column=2)

tk.Label(root, text="Swing View:").grid(row=1, column=0)
swing_view_var = tk.StringVar(value="down_the_line")  # Default value
tk.Radiobutton(root, text="Down the Line", variable=swing_view_var, value="down_the_line").grid(row=1, column=1, sticky="w")
tk.Radiobutton(root, text="Face On", variable=swing_view_var, value="face_on").grid(row=1, column=2, sticky="w")

tk.Label(root, text="Reference Video Path (Optional):").grid(row=2, column=0)
reference_video_path_var = tk.StringVar()
tk.Entry(root, textvariable=reference_video_path_var).grid(row=2, column=1)
tk.Button(root, text="Browse", command=browse_reference_video).grid(row=2, column=2)

tk.Label(root, text="Show Landmarks:").grid(row=3, column=0)
toggle_landmarks_var = tk.BooleanVar(value=True)  # Default is False
tk.Radiobutton(root, text="Yes", variable=toggle_landmarks_var, value=True).grid(row=3, column=1, sticky="w")
tk.Radiobutton(root, text="No", variable=toggle_landmarks_var, value=False).grid(row=3, column=2, sticky="w")

tk.Label(root, text="Save Image:").grid(row=4, column=0)
save_image_var = tk.BooleanVar(value=False)  # Default is False
tk.Radiobutton(root, text="Yes", variable=save_image_var, value=True).grid(row=4, column=1, sticky="w")
tk.Radiobutton(root, text="No", variable=save_image_var, value=False).grid(row=4, column=2, sticky="w")

tk.Label(root, text="Swing Event:").grid(row=5, column=0)
swing_event_var = tk.StringVar(value="Full swing")  # Default value
swing_events = ['Full swing', 'Address', 'Toe-up', 'Mid-backswing', 'Top', 'Mid-downswing', 'Impact', 'Mid-follow-through', 'Finish']
tk.OptionMenu(root, swing_event_var, *swing_events).grid(row=5, column=1, columnspan=2)

tk.Label(root, text="Number of Recommendations:").grid(row=6, column=0)
n_recommendations_var = tk.StringVar(value="5")
tk.Entry(root, textvariable=n_recommendations_var).grid(row=6, column=1)

tk.Button(root, text="Run Analysis", command=run_analysis).grid(row=7, column=1)

# Text widget to display recommendations for A
tk.Label(root, text="Consider adjusting the following:").grid(row=8, column=0)
recommendations_output_1 = tk.Text(root, height=5, width=50)
recommendations_output_1.grid(row=8, column=1, columnspan=2)

# Text widget to display recommendations for B
tk.Label(root, text="Swing events to focus on").grid(row=9, column=0)
recommendations_output_2 = tk.Text(root, height=5, width=50)
recommendations_output_2.grid(row=9, column=1, columnspan=2)

# Text widget to display recommendations for C
tk.Label(root, text="Body parts to focus on").grid(row=10, column=0)
recommendations_output_3 = tk.Text(root, height=5, width=50)
recommendations_output_3.grid(row=10, column=1, columnspan=2)


# Run the application
root.mainloop()