# NOTE: Updated on 02.23.2025
import os, glob, json, random
import pandas as pd

from itertools import groupby
from moviepy import *

ava_action_of_interest_dict = { 
    17: "carry/hold (an object)",
    22: "close (e.g., a door, a box)",
    24: "cut",
    36: "lift/pick up",
    38: "open (e.g., a window, a car door)",
    44: "press",
    45: "pull (an object)",
    46: "push (an object)",
    47: "put down",
    60: "turn (e.g., a screwdriver)",
}

class AVA():
    def __init__(self, annotation_file_path="ava_v2.2/ava_train_v2.2.csv", action_of_interest_dict=ava_action_of_interest_dict):
        columns = ["video_id", "timestamp", "x1", "y1", "x2", "y2", "action_id", "person_id"]
        self.df_annotation = pd.read_csv(annotation_file_path, header=None, names=columns)
        # self.df_annotation.columns = [
        #     "video_id", "timestamp", "x1", "y1", "x2", "y2", "action_id", "person_id"
        # ]
        self.action_of_interest_dict = action_of_interest_dict
        
        self.df_interest = self.df_annotation[self.df_annotation["action_id"].isin(action_of_interest_dict.keys())]
        self.video_filenames = self.df_interest["video_id"].unique()
        self.video_timestamps = self.df_interest.groupby("video_id")["timestamp"].apply(list) # .to_dict()

    def filter_continuous_segments(self, timestamps, min_continous_length=2): # unit: seconds
        """
        filter continuous segments of actions in the video.
        """
        sorted_times = sorted(timestamps)
        segments = []
        temp_segment = [sorted_times[0]]
        for i in range(1, len(sorted_times)):
            if sorted_times[i] == sorted_times[i - 1] + 1:
                temp_segment.append(sorted_times[i])
            else:
                if len(temp_segment) >= min_continous_length:
                    segments.append(temp_segment[:])
                temp_segment = [sorted_times[i]]

        if len(temp_segment) >= min_continous_length:
            segments.append(temp_segment)

        return [t for segment in segments for t in segment]
    
    def filter_videos_with_continuous_segments(self):
        """
        Filter videos with continuous segments of actions.
        """
        
        filtered_video_timestamps = self.video_timestamps.apply(self.filter_continuous_segments)
        filtered_video_timestamps = filtered_video_timestamps[filtered_video_timestamps.apply(len) > 0]
        self.filtered_video_timestamps = filtered_video_timestamps # .to_dict()
        print(f"Filtered videos: {len(self.filtered_video_timestamps)}")
        return self.filtered_video_timestamps
    
    def extract_video_clip(self, video_folder_path="./ava/train", output_folder_path="./ava/train_preprocessed", duration=80, start_padding=10, min_continous_length=20):
        """
        Extract video clip from the original video.
        NOTE: Duration is set to 80 secs by default, because the average duration of VEHS videos is 80 secs.
        NOTE: If duration is set to 120 secs, it will cover about the 80% of the VEHS videos.
        NOTE: If duration is set to 140 secs, it will cover about the 85% of the VEHS videos.
        start_padding is set to 10 secs by default, meaning the extracted video starts 10 seconds earlier.
        """
        
        def find_continuous_segments(timestamps, min_continous_length=min_continous_length):
            sorted_times = sorted(timestamps)
            segments = []
            
            for k, g in groupby(enumerate(sorted_times), lambda ix: ix[0] - ix[1]):
                group = list(g)
                start = group[0][1]
                end = group[-1][1]
                if end - start >= min_continous_length:
                    segments.append((start, end))
            
            return segments

        video_path_dict = {os.path.splitext(os.path.basename(f))[0]: f for f in glob.glob(os.path.join(video_folder_path, "*"))}
        os.makedirs(output_folder_path, exist_ok=True)

        clip_info_path = os.path.join(output_folder_path, "clip_info.csv")
        clip_info = []

        video_segments = {}
        for video_id, times in self.filtered_video_timestamps.items():
            segments = find_continuous_segments(times, min_continous_length=min_continous_length)
            if segments:
                video_segments[video_id] = segments
        self.video_segments = video_segments
        
        start_padding = 10
        print(f"Extracting clips from {len(video_segments)} videos...")
        # receive input from the user (yes or no) to start the extraction process.
        # if input is no, then return.
        # if input is yes, then continue.
        # if input is not yes or no, then ask again.
        user_input = input("Do you want to start the extraction process? (y/n): ")
        while user_input not in ["y", "n"]:
            user_input = input("Please enter 'y' or 'n': ")
        if user_input == "n":
            return
        elif user_input == "y":
            pass

        for video_id, segments in video_segments.items():
            video_path = video_path_dict.get(video_id)
            
            if not os.path.exists(video_path):
                print(f"I can't find the video: {video_path}")
                continue
            
            start_time = segments[random.randint(0, len(segments) - 1)][0] - start_padding # bring a random segment.
            # start_time = segments[0][0] - start_padding # bring the first segment.
            output_path = os.path.join(output_folder_path, f"{video_id}_{start_time}_clip.mp4")

            try:
                with VideoFileClip(video_path) as video:
                    end_time = start_time + duration
                    clip = video.subclipped(start_time, end_time)
                    clip.write_videofile(output_path, codec="libx264")
                    clip_info.append([video_id, start_time, start_time+duration, output_path])
            except Exception as e:
                print(f"extraction error in {video_id}: {e}")
            
        # Save clip info in a csv file
        df_clip_info = pd.DataFrame(clip_info, columns=["video_id", "start_time", "end_time", "clip_path"])
        df_clip_info.to_csv(clip_info_path, index=False)

    def annotate_clips(self, clip_inpo_path="./ava/train_preprocessed/clip_info.csv", output_path="./ava/train_preprocessed/annotation.json"):
        df_clip_info = pd.read_csv(clip_inpo_path)
        annotation_dict = {}

        for i in range(len(df_clip_info)):
            video_id = df_clip_info.loc[i, "video_id"]
            start_time = df_clip_info.loc[i, "start_time"]
            end_time = df_clip_info.loc[i, "end_time"]
            clip_path = df_clip_info.loc[i, "clip_path"]

            df_clip_annotation = self.df_annotation[(self.df_annotation["video_id"] == video_id) & (self.df_annotation["timestamp"] >= start_time) & (self.df_annotation["timestamp"] <= end_time)]
            annotated_seconds = df_clip_annotation["timestamp"].unique()

            second_actions_dict = {}
            for sec in range(start_time, end_time + 1):
                second_actions_dict[sec] = []
                if sec not in annotated_seconds:
                    second_actions_dict[sec].append("none")
                else:
                    unique_actions = df_clip_annotation[df_clip_annotation["timestamp"] == sec]["action_id"].unique()
                    for unique_action_id in unique_actions:
                        if unique_action_id not in self.action_of_interest_dict:
                            second_actions_dict[sec].append("none")
                        else:
                            second_actions_dict[sec].append(self.action_of_interest_dict[unique_action_id])
                    
                second_actions_dict[sec] = list(set(second_actions_dict[sec]))

            annotation_dict[clip_path] = second_actions_dict

        with open(output_path, "w") as f:
            json.dump(annotation_dict, f)
        
        return

if __name__ == "__main__":
    ava = AVA(annotation_file_path="ava_v2.2/ava_train_v2.2.csv", action_of_interest_dict=ava_action_of_interest_dict)
    _ = ava.filter_videos_with_continuous_segments()
    ava.extract_video_clip(start_time=0)
    ava.annotate_clips()
