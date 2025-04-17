import os, glob, json
import re
import base64, cv2
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
# import moviepy.video.fx.all as vfx

from datetime import datetime
from collections import Counter
from moviepy import VideoFileClip

def extract_text_from_tags(response, tag_name="action"):
    # match = re.search(rf"<{tag_name}>(.*?)</{tag_name}>", response)
    return re.findall(rf"<{tag_name}>(.*?)</{tag_name}>", response)

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")
    
def extract_frames(video_path, fps, output_dir=".", save=False):
    """
    Extract frames from a video file at a specified frames per second (fps) rate and optionally saves them.
    Args:
        video_path (str): Path to the input video file.
        fps (int): Target frames per second to extract from the video.
        output_dir (str, optional): Directory to save the extracted frames. Defaults to ".".
        save (bool, optional): Whether to save the extracted frames to disk. Defaults to False.
    Returns:
        dict: A dictionary where keys are seconds (int) and values are lists of base64-encoded frame images (str).
    Raises:
        ValueError: If the video file cannot be opened.
    """
    
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    save_dir = os.path.join(output_dir, video_name)
    
    if save:
        os.makedirs(save_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Unable to open video file {video_path}")
        return
    original_fps = cap.get(cv2.CAP_PROP_FPS)

    frames_by_second = {}
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        seconds = int(frame_count / original_fps)
        frame_time = frame_count / original_fps
        filename = f"frame_{frame_count}_{frame_time:.2f}sec.jpg"
        
        if seconds not in frames_by_second:
            frames_by_second[seconds] = []

        _, buffer = cv2.imencode(".jpg", frame)
        encoded_frame = base64.b64encode(buffer).decode("utf-8")
        frames_by_second[seconds].append((filename, frame, encoded_frame))
        
        frame_count += 1
    
    cap.release()

    sampled_frames_by_second = {}
    for second, frame_tuples in sorted(frames_by_second.items(), key=lambda x: int(x[0])):
        if len(frame_tuples) <= fps:
            # if the number of frames is less than the target fps, use all frames
            sampled_indices = list(range(len(frame_tuples)))
        else:
            if fps == 1:
                # if the target fps is 1, use the middle frame
                sampled_indices = [len(frame_tuples) // 2]
            else:
                # otherwise, sample frames evenly
                step = len(frame_tuples) / (fps - 1)
                sampled_indices = [round(i * step) for i in range(fps)]
        
        sampled_indices = list(set(sampled_indices))  # Ensure unique indices
        sampled_indices = [min(len(frame_tuples) - 1, idx) for idx in sampled_indices]  # Clamp indices to valid range
        sampled_encoded_frames = []

        for idx in sampled_indices:
            if save:
                frame_filename = os.path.join(save_dir, frame_tuples[idx][0])
                cv2.imwrite(frame_filename, frame_tuples[idx][1])
            sampled_encoded_frames.append(frame_tuples[idx][2]) # NOTE: if you want to check the frame, use frame_tuples[idx][0]
        sampled_frames_by_second[second] = sampled_encoded_frames
    
    return sampled_frames_by_second

def extract_frames_for_gemini(video_path, fps, output_dir="./temp_gemini"):
    """
    Extract frames from a video file at a specified frames per second (fps) rate and optionally saves them.
    Args:
        video_path (str): Path to the input video file.
        fps (int): Target frames per second to extract from the video.
        output_dir (str, optional): Directory to save the extracted frames. Defaults to ".".
        save (bool, optional): Whether to save the extracted frames to disk. Defaults to False.
    Returns:
        dict: A dictionary where keys are seconds (int) and values are lists of frame paths (str).
    Raises:
        ValueError: If the video file cannot be opened.
    """
    
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    save_dir = os.path.join(output_dir, video_name)
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(save_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Unable to open video file {video_path}")
        return
    original_fps = cap.get(cv2.CAP_PROP_FPS)

    frames_by_second = {}
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        seconds = int(frame_count / original_fps)
        frame_time = frame_count / original_fps
        filename = f"frame_{frame_count}_{frame_time:.2f}sec.jpg"
        
        if seconds not in frames_by_second:
            frames_by_second[seconds] = []

        frames_by_second[seconds].append((filename, frame))
        
        frame_count += 1
    
    cap.release()

    sampled_frames_by_second = {}
    for second, frame_tuples in sorted(frames_by_second.items(), key=lambda x: int(x[0])):
        if len(frame_tuples) <= fps:
            # if the number of frames is less than the target fps, use all frames
            sampled_indices = list(range(len(frame_tuples)))
        else:
            if fps == 1:
                # if the target fps is 1, use the middle frame
                sampled_indices = [len(frame_tuples) // 2]
            else:
                # otherwise, sample frames evenly
                step = len(frame_tuples) / (fps - 1)
                sampled_indices = [round(i * step) for i in range(fps)]
        
        sampled_indices = list(set(sampled_indices))  # Ensure unique indices
        sampled_indices = [min(len(frame_tuples) - 1, idx) for idx in sampled_indices]  # Clamp indices to valid range
        sampled_frames = []

        for idx in sampled_indices:
            frame_filename = os.path.join(save_dir, frame_tuples[idx][0])
            cv2.imwrite(frame_filename, frame_tuples[idx][1])
            sampled_frames.append(frame_filename)
        sampled_frames_by_second[second] = sampled_frames
    
    return sampled_frames_by_second

def extend_video(video_path, slow_factor):
    # In Google Gemini, slow_factor will be the same as fps.
    # cap = cv2.VideoCapture(video_path)
    
    # fps = int(cap.get(cv2.CAP_PROP_FPS))
    # width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    # height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    # fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    
    # new_fps = fps / slow_factor
    
    # out = cv2.VideoWriter(f"temp_{slow_factor}.mp4", fourcc, new_fps, (width, height))
    
    # while cap.isOpened():
    #     ret, frame = cap.read()
    #     if not ret:
    #         break
    #     out.write(frame)
    
    # cap.release()
    # out.release()
    clip = VideoFileClip(video_path)
    slowed_clip = clip.with_speed_scaled(factor=1/slow_factor)
    slowed_clip.write_videofile(f"temp_{slow_factor}.mp4", fps=clip.fps)
    return
                            
def resize_video(video_path, output_path, max_size=2000):
    try:
        with VideoFileClip(video_path) as video:
            width, height = video.size  # current video size

            # if the video is larger than the max_size, resize it
            if width > max_size or height > max_size:
                scale_factor = min(max_size / width, max_size / height)  # maintain aspect ratio
                new_width = int(width * scale_factor)
                new_height = int(height * scale_factor)

                resized_video = video.resized((new_width, new_height))
                resized_video.write_videofile(output_path, codec="libx264")
                return output_path
            else:
                return video_path

    except Exception as e:
        print(f"Video resizing error: {e}")
        return video_path

def get_video_duration(video_path):
    try:
        with VideoFileClip(video_path) as video:
            return int(video.duration)
    except Exception as e:
        print(f"Error occurred when processing {video_path}: {e}")
        return None

def get_majority_vote(predictions):
    counter = Counter(predictions)
    return counter.most_common(1)[0][0]

def print_output(df_output):
    """
    Print the output in a human-readable format.
    Args:
        df_output (pd.DataFrame): A DataFrame containing the seconds and predictions.
    """

    for i in range(len(df_output)):
        if i == 0:
            if df_output.iloc[i]["predictions"] != "none":
                print(f"Action: {df_output.iloc[i]['predictions']} | Start Time: {df_output.iloc[i]['seconds']}s", end=" | ")
        else:
            if df_output.iloc[i]["predictions"] != df_output.iloc[i-1]["predictions"]:
                if df_output.iloc[i]["predictions"] != "none":
                    print(f"End Time: {df_output.iloc[i]['seconds']}s")
                    print(f"Action: {df_output.iloc[i]['predictions']} | Start Time: {df_output.iloc[i]['seconds']}s", end=" | ")
                else:
                    pass
            else:
                pass
    print(f"End Time: {df_output.iloc[-1]['seconds']}s")
    return

def visualize_output(predicted_actions, actual_actions=None, save_path=None, legend_classes=None, second_width=1, figsize=(15, 3)):
    color_map = {
        action: ("gray" if action == "none" else plt.cm.tab20(i / len(legend_classes)))
        for i, action in enumerate(legend_classes)
    }
    num_seconds = len(predicted_actions)
    
    time_ticks = range(0, num_seconds * second_width, second_width)
    if actual_actions is None:
        fig, ax = plt.subplots(1, 1, figsize=figsize, sharex=True)

        # bar: predicted actions
        ax.bar(
            time_ticks,
            [1] * num_seconds,
            width=second_width,
            align="edge",
            color=[color_map[p] for p in predicted_actions],
            edgecolor="none",
        )
        ax.set_title("Predicted Actions (Second-wise)")
        ax.set_yticks([])
        ax.set_xlim(0, num_seconds)

        # time axis settings
        ax.set_xticks(np.arange(0, num_seconds+1, 1))
        ax.set_xlabel("Time (s)")
        
        # legend generation
        legend_patches = [
            mpatches.Patch(color=color_map[action], label=action) for action in legend_classes
        ]

        fig.legend(
            handles=legend_patches,
            loc="upper center",
            ncol=len(legend_classes),
            bbox_to_anchor=(0.5, 1.25)
        )

        plt.tight_layout(rect=[0, 0, 1, 0.95])  # legend space adjustment
        plt.show()
    else:
        if len(actual_actions) < len(predicted_actions):
            predicted_actions = predicted_actions[:len(actual_actions)]
            print("The number of actual actions is less than the number of predicted actions. Truncating the predicted actions.")
        elif len(actual_actions) > len(predicted_actions):
            actual_actions = actual_actions[:len(predicted_actions)]
            print("The number of predicted actions is less than the number of actual actions. Truncating the actual actions.")

        # assert len(actual_actions) == len(predicted_actions), "The number of actual and predicted actions must be the same."
        fig, ax = plt.subplots(2, 1, figsize=figsize, sharex=True)

        # upper bar: actual actions
        ax[0].bar(
            time_ticks,
            [1] * num_seconds,
            width=second_width,
            align="edge",
            color=[color_map[a] for a in actual_actions],
            edgecolor="none",
        )
        ax[0].set_title("Actual Actions (Second-wise)")
        ax[0].set_yticks([])
        ax[0].set_xlim(0, num_seconds)

        # bottom bar: predicted actions
        ax[1].bar(
            time_ticks,
            [1] * num_seconds,
            width=second_width,
            align="edge",
            color=[color_map[p] for p in predicted_actions],
            edgecolor="none",
        )
        ax[1].set_title("Predicted Actions (Second-wise)")
        ax[1].set_yticks([])
        ax[1].set_xlim(0, num_seconds)

        # time axis settings
        ax[1].set_xticks(np.arange(0, num_seconds+1, 1))
        ax[1].set_xlabel("Time (s)")
        
        # legend generation
        legend_patches = [
            mpatches.Patch(color=color_map[action], label=action) for action in legend_classes
        ]

        fig.legend(
            handles=legend_patches,
            loc="upper center",
            ncol=len(legend_classes),
            bbox_to_anchor=(0.5, 1.25)
        )

        plt.tight_layout(rect=[0, 0, 1, 0.95])  # legend space adjustment
        # plt.show()
        if save_path is not None:
            plt.savefig(save_path)
    return

def fill_missing_seconds(seconds, predictions, duration):
    if len(seconds) == duration + 1:
        return seconds, predictions
    
    if abs(len(seconds)-duration) != 1:
        print(f"Error: The number of seconds is very different from the duration of the video.")

    corrected_seconds = []
    corrected_predictions = []
    
    for t in range(duration + 1):
        if t in seconds:
            idx = seconds.index(t)
            corrected_seconds.append(t)
            corrected_predictions.append(predictions[idx])
        else:
            if corrected_predictions:
                corrected_predictions.append(corrected_predictions[-1])
            else:
                corrected_predictions.append("none")
            
            corrected_seconds.append(t)    
    return corrected_seconds, corrected_predictions

def append_df_to_json(df_prediction, video_path, json_path):
    # video_filename = os.path.basename(video_path).split(".mp4")[0]
    seconds = df_prediction["seconds"].tolist()
    predictions = df_prediction["predictions"].tolist()
    update_dict = {video_path: {}}
    for second, prediction in zip(seconds, predictions):
        update_dict[video_path][second] = prediction
    
    # check if the json file exists
    if os.path.exists(json_path):
        with open(json_path, "r") as f:
            data = json.load(f)
            # stack the new data on bottom of the existing data
            data.update(update_dict)
    else:        
        data = update_dict
    
    # save the updated json
    with open(json_path, "w") as f:
        json.dump(data, f)
    return

def set_timecodes_func_for_gemini(timecodes):
    return [{**t, "text": t["text"].replace("\\'", "'")} for t in timecodes]

def set_timecodes_with_objects_func_for_gemini(timecodes):
    return [{**t, "text": t["text"].replace("\\'", "'")} for t in timecodes]

def set_timecodes_with_descriptions_func_for_gemini(timecodes):
    return [{**t, "text": t["spoken_text"].replace("\\'", "'")} for t in timecodes]

def fix_broken_json(txt):
    txt = txt.strip()
    
    match_text = re.search(r'"text": "(.*?)$', txt.strip())
    match_time = re.search(r'"time": "(.*?)$', txt.strip())
    match_broken_entry = re.search(r'\{\s*"time":\s*("[^"]*")?\s*(,\s*"text":\s*("[^"]*")?)?\s*$', txt)
    match_broken_entry_2 = re.search(r',?\s*\{\s*"time":\s*"[^"]*"\s*(,\s*"text":\s*"[^"]*")?\s*,?\s*$', txt)
    if match_broken_entry or match_text or match_time or match_broken_entry_2:
        txt = re.sub(r',?\s*\{\s*"time":\s*("[^"]*")?\s*(,\s*"text":\s*("[^"]*")?)?\s*$', '', txt.strip())
        txt = re.sub(r',?\s*\{\s*"time":\s*"[^"]*"\s*(,\s*"text":\s*"[^"]*")?\s*,?\s*$', '', txt.strip())
        txt = re.sub(r',?\s*{\s*"time":\s*".*?",?\s*"text":\s*".*?$', '', txt.strip())


    if not txt.endswith("]"):
        txt += "]"

    return txt

def time_to_seconds(time_str):
    exist_end_time = False
    # if time_str is in the format 00:00-00:08, take the first time
    if "-" in time_str:
        exist_end_time = True
        
        start_time_str = time_str.split("-")[0]
        start_time_obj = datetime.strptime(start_time_str, "%M:%S")
        end_time_str = time_str.split("-")[1]
        end_time_obj = datetime.strptime(end_time_str, "%M:%S")
        return start_time_obj.minute * 60 + start_time_obj.second, end_time_obj.minute * 60 + end_time_obj.second, exist_end_time
    else:
        time_obj = datetime.strptime(time_str, "%M:%S")
        return time_obj.minute * 60 + time_obj.second, None, exist_end_time
    
def load_json(json_path):
    if os.path.exists(json_path):
        with open(json_path, "r") as f:
            data = json.load(f)
            return data
    else:
        print(f"Error: The file {json_path} does not exist.")
        return None