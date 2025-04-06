import os, glob, time, json, re, random
import base64, cv2
import pandas as pd

from datetime import datetime
from transformers import BertTokenizer, BertModel
import torch
import torch.nn.functional as F

from openai import OpenAI
import anthropic
# import google.generativeai as genai #NOTE: previous version
# from google.generativeai import types #NOTE: previous version

from google import genai
from google.genai import types

from twelvelabs import TwelveLabs
from twelvelabs.models.task import Task
from twelvelabs import APIStatusError

import utils, prompting

api_pricing_dict = {
    "gpt-4o": {"input_cost_per_1Mtks": 2.5, "cache_cost_per_1Mtks": 1.25, "output_cost_per_1Mtks": 10.0},
    "gpt-4o-mini": {"input_cost_per_1Mtks": 0.15, "cache_cost_per_1Mtks": 0.075, "output_cost_per_1Mtks": 0.6},
    "o1": {"input_cost_per_1Mtks": 15.0, "cache_cost_per_1Mtks": 7.50, "output_cost_per_1Mtks": 60.0},
    "claude-3-5-sonnet-latest": {"input_cost_per_1Mtks": 3.0, "cache_cost_per_1Mtks": 3.75, "output_cost_per_1Mtks": 15.0},
    "gemini-2.0-flash-exp": {"input_cost_per_1Mtks": 1.25, "cache_cost_per_1Mtks": 0.00, "output_cost_per_1Mtks": 5.0},
    "gemini-2.0-flash": {"input_cost_per_1Mtks": 0.1, "cache_cost_per_1Mtks": 0.4, "output_cost_per_1Mtks": 1.0},
    "gemini-1.5-pro": {"input_cost_per_1Mtks": 1.25, "cache_cost_per_1Mtks": 0.3125, "output_cost_per_1Mtks": 5.0},
    "pegasus1.2": {"input_cost_per_1Mtks": "Please use `input_cost_per_1min`.", "input_cost_per_1min":0.0413, "cache_cost_per_1Mtks": 0.0, "output_cost_per_1Mtks": 2.0},
}

class AgentOpenAI():
    def __init__(self, logger=None, model_name="gpt-4o", api_key=None, region=None):
        """
        Initialize OpenAI Agent class.
        Args:
            logger (Logger): Logger instance for logging information. NOTE: Unused.
            model_name (str): Name of the model to use. Defaults to "gpt-4o".
            api_key (str): API key for authentication. Defaults to None.
            region (str, optional): Region for the API. Defaults to None. NOTE: Unused.
        """
        self.logger = logger
        self.model_name = model_name
        self.pricing_dict = api_pricing_dict[model_name]
        self.api_key = api_key
        self.region = region
        self.agent = OpenAI(api_key=api_key)
        self.max_video_size = 3072
        self.reviser = AgentBERT()
    
    def ask_about_video(self, video_path, prompt="", scope=[], fps=1, temperature=0.0, max_tokens=128, output_folder_dir="output", save=False, show=False):        
        scope = [item.lower() for item in scope]
        duration = utils.get_video_duration(video_path)
        if duration is None:
            raise ValueError(f"Unable to get the duration of the video file: {video_path}")
        
        video_path = utils.resize_video(video_path, "temp.mp4", self.max_video_size)
        sampled_frames_by_second = utils.extract_frames(video_path, fps)
        total_cost = 0.0
        start_time = time.time()
        predictions = []
        num_inconsistent_output = 0

        for second, encoded_frames in sampled_frames_by_second.items():
            OPENAI_PROMPT_MESSAGES = [
                {
                    "role": "user",
                    "content": [
                        prompt,
                        *map(lambda x: {"image": x, "resize": 768}, encoded_frames),
                    ],
                },
            ]
            
            params = {
            "model": self.model_name,
            "messages": OPENAI_PROMPT_MESSAGES,
            "max_tokens": max_tokens,
            "temperature": temperature,
            }

            response = self.agent.chat.completions.create(**params)
            prediction = response.choices[0].message.content
            prediction = utils.extract_text_from_tags(prediction, tag_name="action")
            if prediction.lower() not in scope:
                best_text, best_sim = self.reviser.get_similarity(prediction, scope)
                if best_sim > 0.9:
                    prediction = best_text
                else:
                    print(f"Invalid action prediction: {prediction}")
                    prediction = "none"
                num_inconsistent_output += 1
            
            cost = (response.usage.prompt_tokens*self.pricing_dict["input_cost_per_1Mtks"] + response.usage.completion_tokens*self.pricing_dict["output_cost_per_1Mtks"])/1000000
            total_cost += cost

            predictions.append((second, prediction.lower()))
        
        self.inference_time = time.time() - start_time
        self.inference_cost = total_cost
        print(f"Elapsed time: {self.inference_time:.2f} seconds")
        print(f"Total cost: ${self.inference_cost:.2f}")
        if num_inconsistent_output > 0:
            print(f"Ratio of inconsistent outputs: {num_inconsistent_output}/{len(sampled_frames_by_second)}")

        sorted_predictions = sorted(predictions, key=lambda x: x[0])
        seconds = [item[0] for item in sorted_predictions]
        predictions = [item[1] for item in sorted_predictions]
        seconds, predictions = utils.fill_missing_seconds(seconds, predictions, duration)
        output = pd.DataFrame({"seconds": seconds, "predictions": predictions})

        utils.print_output(output)
        if show:
            utils.visualize_output(output["predictions"], save_path=None, legend_classes=scope, second_width=1, figsize=(15, 3))

        if save:
            output.to_csv(os.path.join(output_folder_dir, f"predictions_{self.model_name}.csv"), index=False)
        
        if os.path.exists("temp.mp4"):
            os.remove("temp.mp4")

        return output
    
    def classify_video(self, video_path, prompt="", scope=[], fps=16, temperature=0.0, max_tokens=256):
        sampled_frames_by_second = utils.extract_frames(video_path, 1)
        all_frames = [frames[0] for sec, frames in sorted(sampled_frames_by_second.items(), key=lambda x: int(x[0]))]
        
        sampled_frames = all_frames[::int(len(all_frames)/fps)]
        if len(sampled_frames) < fps:
            print(f"Number of sampled frames: {len(sampled_frames)}")

        OPENAI_PROMPT_MESSAGES = [
                {
                    "role": "user",
                    "content": [
                        prompt,
                        *map(lambda x: {"image": x, "resize": 768}, sampled_frames),
                    ],
                },
            ]
            
        params = {
        "model": self.model_name,
        "messages": OPENAI_PROMPT_MESSAGES,
        "max_tokens": max_tokens,
        "temperature": temperature,
        }

        response = self.agent.chat.completions.create(**params)
        prediction = response.choices[0].message.content
        prediction = utils.extract_text_from_tags(prediction, tag_name="task")
        
        if prediction not in scope:
            best_text, best_sim = self.reviser.get_similarity(prediction, scope)
            prediction = best_text
            print(f"Invalid prediction is changed: {prediction} -> {best_text}")

        return prediction
    
    def ask_questions_about_video(self, video_path, prompts=[""], scope=[], fps=1, temperature=0.0, max_tokens=128, path_gt_second_action_dict="", output_folder_dir="output", save=False, show=False):
        scope = [item.lower() for item in scope]
        duration = utils.get_video_duration(video_path)
        if duration is None:
            raise ValueError(f"Unable to get the duration of the video file: {video_path}")
        
        video_path = utils.resize_video(video_path, "temp.mp4", self.max_video_size)
        sampled_frames_by_second = utils.extract_frames(video_path, fps)
        total_cost = 0.0
        start_time = time.time()
        predictions = []
        num_inconsistent_output = 0
        
        with open(path_gt_second_action_dict, "r") as f:
            gt_second_action_dicts = json.load(f)
        video_folder_dir = "./COIN_videos/filtered_COIN_videos"
        video_path_key = os.path.join(video_folder_dir, os.path.basename(video_path))
        gt_second_action_dicts = gt_second_action_dicts.get(video_path_key, {})
        assert len(gt_second_action_dicts) > 0, f"Unable to get the ground truth action dict for {video_path}"

        for second, encoded_frames in sampled_frames_by_second.items():
            
            # inital prompt
            OPENAI_PROMPT_MESSAGES = [
                    {
                        "role": "user",
                        "content": [
                            prompts[0],
                            *map(lambda x: {"image": x, "resize": 768}, encoded_frames),
                        ],
                    },
                ]
            params = {
                "model": self.model_name,
                "messages": OPENAI_PROMPT_MESSAGES,
                "max_tokens": max_tokens,
                "temperature": temperature,
                }

            response = self.agent.chat.completions.create(**params)
            prediction = response.choices[0].message.content
            prediction = utils.extract_text_from_tags(prediction, tag_name="action")
            if prediction.lower() not in scope:
                best_text, best_sim = self.reviser.get_similarity(prediction, scope)
                if best_sim > 0.9:
                    prediction = best_text
                else:
                    print(f"Invalid action prediction: {prediction}")
                    prediction = "none"
                num_inconsistent_output += 1
            
            cost = (response.usage.prompt_tokens*self.pricing_dict["input_cost_per_1Mtks"] + response.usage.completion_tokens*self.pricing_dict["output_cost_per_1Mtks"])/1000000
            total_cost += cost
            
            if prediction.lower() == gt_second_action_dicts.get(str(second), 'none'):
                predictions.append((second, prediction.lower()))
                continue
            
            OPENAI_PROMPT_MESSAGES.append({"role": "assistant", "content": prediction})
            
            new_prompts = prompts.copy()[1:]
            num_prompts = len(new_prompts)
            # iterate through each prompt and get the response
            while num_prompts > 0:
                prompt = new_prompts.pop(0)
                num_prompts -= 1
                prompt += f"{gt_second_action_dicts.get(str(second), 'none')}"
                OPENAI_PROMPT_MESSAGES.append({"role": "user", "content": prompt})
                            
                params = {
                "model": self.model_name,
                "messages": OPENAI_PROMPT_MESSAGES,
                "max_tokens": max_tokens,
                "temperature": temperature,
                }

                response = self.agent.chat.completions.create(**params)
                prediction = response.choices[0].message.content
                prediction = utils.extract_text_from_tags(prediction, tag_name="action")
                print("second:", second, "prompt:", prompt, "prediction:", prediction)
                if prediction.lower() not in scope:
                    best_text, best_sim = self.reviser.get_similarity(prediction, scope)
                    if best_sim > 0.9:
                        prediction = best_text
                    else:
                        print(f"Invalid action prediction: {prediction}")
                        prediction = "none"
                    num_inconsistent_output += 1

                cost = (response.usage.prompt_tokens*self.pricing_dict["input_cost_per_1Mtks"] + response.usage.completion_tokens*self.pricing_dict["output_cost_per_1Mtks"])/1000000
                total_cost += cost
                OPENAI_PROMPT_MESSAGES.append({"role": "assistant", "content": prediction})

            predictions.append((second, prediction.lower()))
        
        self.inference_time = time.time() - start_time
        self.inference_cost = total_cost
        print(f"Elapsed time: {self.inference_time:.2f} seconds")
        print(f"Total cost: ${self.inference_cost:.2f}")
        if num_inconsistent_output > 0:
            print(f"Ratio of inconsistent outputs: {num_inconsistent_output}/{len(sampled_frames_by_second)}")

        sorted_predictions = sorted(predictions, key=lambda x: x[0])
        seconds = [item[0] for item in sorted_predictions]
        predictions = [item[1] for item in sorted_predictions]
        seconds, predictions = utils.fill_missing_seconds(seconds, predictions, duration)
        output = pd.DataFrame({"seconds": seconds, "predictions": predictions})

        utils.print_output(output)
        if show:
            utils.visualize_output(output["predictions"], save_path=None, legend_classes=scope, second_width=1, figsize=(15, 3))

        if save:
            output.to_csv(os.path.join(output_folder_dir, f"predictions_{self.model_name}.csv"), index=False)
        
        if os.path.exists("temp.mp4"):
            os.remove("temp.mp4")
        
        return output

class AgentAnthropic():
    def __init__(self, logger=None, model_name="claude-3-5-sonnet-latest", api_key=None, region=None):
        """
        Initialize Anthropic Agent class.
        Args:
            logger (Logger): Logger instance for logging information. NOTE: Unused.
            model_name (str): Name of the model to use. Defaults to "claude-3-5-sonnet-latest".
            api_key (str): API key for authentication. Defaults to None.
            region (str, optional): Region for the API. Defaults to None. NOTE: Unused.
        """
        self.logger = logger
        self.model_name = model_name
        self.pricing_dict = api_pricing_dict[model_name]
        self.api_key = api_key
        self.region = region
        self.agent = anthropic.Anthropic(api_key=api_key)
        self.max_video_size = 8000
        self.reviser = AgentBERT()
    
    def ask_about_video(self, video_path, prompt="", scope=[], fps=1, temperature=0.0, max_tokens=128, output_folder_dir="output", save=False, show=False):
        scope = [item.lower() for item in scope]
        duration = utils.get_video_duration(video_path)
        if duration is None:
            raise ValueError(f"Unable to get the duration of the video file: {video_path}")
        
        video_path = utils.resize_video(video_path, "temp.mp4", self.max_video_size)
        sampled_frames_by_second = utils.extract_frames(video_path, fps)
        total_cost = 0.0
        start_time = time.time()
        predictions = []
        num_inconsistent_output = 0
        
        ANTHROPIC_PROMPT_MESSAGES=[
            {
                "role": 'user',
                "content": None
            }
        ]

        for second, encoded_frames in sampled_frames_by_second.items():
            MESSAGES_PER_SECOND = ANTHROPIC_PROMPT_MESSAGES.copy()
            vision_content = []
            for i, encoded_frame in enumerate(encoded_frames):
                vision_content.append(
                    {"type": "text", "text": f"Frame {i+1}:"}
                )
                vision_content.append(
                    {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": encoded_frame}}
                )
                vision_content.append(
                    {"type": "text", "text": prompt}
                )
            MESSAGES_PER_SECOND[-1]["content"] = vision_content

            response = self.agent.messages.create(
                model=self.model_name,
                max_tokens=max_tokens,
                messages=MESSAGES_PER_SECOND, 
                temperature=0
            )
            prediction = response.content[0].text
            prediction = utils.extract_text_from_tags(prediction, tag_name="action")

            if prediction.lower() not in scope:
                best_text, best_sim = self.reviser.get_similarity(prediction, scope)
                if best_sim > 0.9:
                    prediction = best_text
                else:
                    print(f"Invalid action prediction: {prediction}")
                    prediction = "none"
                num_inconsistent_output += 1
            
            response_json= json.loads(response.json())
            cost = (response_json["usage"]["input_tokens"]*self.pricing_dict["input_cost_per_1Mtks"] + response_json["usage"]["output_tokens"]*self.pricing_dict["output_cost_per_1Mtks"])/1000000
            total_cost += cost

            predictions.append((second, prediction.lower()))
        
        self.inference_time = time.time() - start_time
        self.inference_cost = total_cost
        
        print(f"Elapsed time: {self.inference_time:.2f} seconds")
        print(f"Total cost: ${self.inference_cost:.2f}")
        if num_inconsistent_output > 0:
            print(f"Ratio of inconsistent outputs: {num_inconsistent_output}/{len(sampled_frames_by_second)}")

        sorted_predictions = sorted(predictions, key=lambda x: x[0])
        seconds = [item[0] for item in sorted_predictions]
        predictions = [item[1] for item in sorted_predictions]
        seconds, predictions = utils.fill_missing_seconds(seconds, predictions, duration)

        output = pd.DataFrame({"seconds": seconds, "predictions": predictions})
        utils.print_output(output)
        if show:
            utils.visualize_output(output["predictions"], save_path=None, legend_classes=action_clusters, second_width=1, figsize=(15, 3))

        if save:
            output.to_csv(os.path.join(output_folder_dir, f"predictions_{self.model_name}.csv"), index=False)
        
        if os.path.exists("temp.mp4"):
            os.remove("temp.mp4")

        return output
    
class AgentGoogle():
    def __init__(self, logger=None, model_name="gemini-2.0-flash-exp", api_key=None, region=None):
        """
        Initialize Google Agent class.
        Args:
            logger (Logger): Logger instance for logging information. NOTE: Unused.
            model_name (str): Name of the model to use. Defaults to "gemini-2.0-flash-exp".
            api_key (str): API key for authentication. Defaults to None.
            region (str, optional): Region for the API. Defaults to None. NOTE: Unused.
        """
        self.logger = logger
        self.model_name = model_name
        self.pricing_dict = api_pricing_dict[model_name]
        self.api_key = api_key
        self.region = region
        self.max_video_size = 2000
        self.reviser = AgentBERT()

        self.SYSTEM_PROMPT = "When given a video and a query, call the relevant function only once with the appropriate timecodes and text for the video"
        
        set_timecodes = types.FunctionDeclaration(
            name="set_timecodes",
            description="Set the timecodes for the video with associated text",
            parameters={
                "type": "OBJECT",
                "properties": {
                    "timecodes": {
                        "type": "ARRAY",
                        "items": {
                            "type": "OBJECT",
                            "properties": {
                                "time": {"type": "STRING"},
                                "text": {"type": "STRING"},
                            },
                            "required": ["time", "text"],
                        }
                    }
                },
                "required": ["timecodes"]
            }
        )

        set_timecodes_with_objects = types.FunctionDeclaration(
            name="set_timecodes_with_objects",
            description="Set the timecodes for the video with associated text and object list",
            parameters={
                "type": "OBJECT",
                "properties": {
                    "timecodes": {
                        "type": "ARRAY",
                        "items": {
                            "type": "OBJECT",
                            "properties": {
                                "time": {"type": "STRING"},
                                "text": {"type": "STRING"},
                                "objects": {
                                    "type": "ARRAY",
                                    "items": {"type": "STRING"},
                                },
                            },
                            "required": ["time", "text", "objects"],
                        }
                    }
                },
                "required": ["timecodes"],
            }
        )

        set_timecodes_with_numeric_values = types.FunctionDeclaration(
            name="set_timecodes_with_numeric_values",
            description="Set the timecodes for the video with associated numeric values",
            parameters={
                "type": "OBJECT",
                "properties": {
                    "timecodes": {
                        "type": "ARRAY",
                        "items": {
                            "type": "OBJECT",
                            "properties": {
                                "time": {"type": "STRING"},
                                "value": {"type": "NUMBER"},
                            },
                            "required": ["time", "value"],
                        }
                    }
                },
                "required": ["timecodes"],
            }
        )

        set_timecodes_with_descriptions = types.FunctionDeclaration(
            name="set_timecodes_with_descriptions",
            description="Set the timecodes for the video with associated spoken text and visual descriptions",
            parameters={
                "type": "OBJECT",
                "properties": {
                    "timecodes": {
                        "type": "ARRAY",
                        "items": {
                            "type": "OBJECT",
                            "properties": {
                                "time": {"type": "STRING"},
                                "spoken_text": {"type": "STRING"},
                                "visual_description": {"type": "STRING"},
                            },
                            "required": ["time", "spoken_text", "visual_description"],
                        }
                    }
                },
                "required": ["timecodes"]
            }
        )

        self.response_schema = {
            "type": "ARRAY",
            "items": {
                "type": "OBJECT",
                "properties": {
                    "time": {"type": "STRING"},
                    "text": {"type": "STRING"},
                },
                "required": ["time", "text"],
            }
        }

        self.VIDEO_TOOLS = types.Tool(function_declarations=[
            set_timecodes, set_timecodes_with_objects, set_timecodes_with_numeric_values, set_timecodes_with_descriptions
            ])
        
        self.agent = genai.Client(api_key=api_key)
        # self.agent = genai.GenerativeModel(model_name, system_instruction=self.SYSTEM_PROMPT, tools=self.VIDEO_TOOLS)  # NOTE: previous version
        # self.agent = genai.GenerativeModel(model_name)                                                                 # NOTE: previous version
        # genai.configure(api_key=api_key)                                                                               # NOTE: previous version
    
    def ask_about_video(self, video_path, prompt="", scope=[], fps=1, temperature=0.0, max_tokens=512, output_folder_dir="output", save=False, show=False):
        scope = [item.lower() for item in scope]
        video_path = utils.resize_video(video_path, "temp.mp4", self.max_video_size)
        original_duration = int(utils.get_video_duration(video_path))
        if fps != 1:
            utils.extend_video(video_path, fps)
            video_path = f"temp_{fps}.mp4"
        duration = int(utils.get_video_duration(video_path))
        
        video_file = self.agent.files.upload(file=video_path)
        # video_file = genai.upload_file(path=video_path) #NOTE: previous version

        # Check whether the file is ready to be used.
        while video_file.state.name == "PROCESSING":
            print('.', end='')
            time.sleep(1)
            video_file = self.agent.files.get(name=video_file.name)
            # video_file = genai.get_file(video_file.name) #NOTE: previous version

        if video_file.state.name == "FAILED":
            raise ValueError(video_file.state.name)

        total_cost = 0.0
        inference_start_time = time.time()
        num_inconsistent_output = 0

        ###############################################################
        # response = self.agent.generate_content( #NOTE: previous version
        #     contents=[video_file, GeminiAnalysisMode['ACT_DETECTION']],

        # )
        
        response = self.agent.models.generate_content(
            model=self.model_name,
            config=types.GenerateContentConfig(
                system_instruction=self.SYSTEM_PROMPT,
                response_mime_type="application/json",
                response_schema=self.response_schema,
                max_output_tokens=max_tokens,
                temperature=temperature),
            contents=[video_file, prompt]
        )
        ###############################################################
        response_text = response.candidates[0].content.parts[0].text
        response_text = utils.fix_broken_json(response_text)
        response_json = json.loads(response_text)
        modified_response = response_json
        """
        # NOTE: This function does not work well.
        if response.candidates[0].content.parts[0].function_call.name == "set_timecodes":
            modified_response = utils.set_timecodes_func_for_gemini(response.candidates[0].content.parts[0].function_call.args['timecodes'])
        elif response.candidates[0].content.parts[0].function_call.name == "set_timecodes_with_objects":
            modified_response = utils.set_timecodes_with_objects_func_for_gemini(response.candidates[0].content.parts[0].function_call.args['timecodes'])
        elif response.candidates[0].content.parts[0].function_call.name == "set_timecodes_with_descriptions":
            modified_response = utils.set_timecodes_with_descriptions_func_for_gemini(response.candidates[0].content.parts[0].function_call.args['timecodes'])
        else:
            raise ValueError("Invalid function call")
        """
        
        predictions = ["none"] * duration

        for i in range(len(modified_response) - 1):
            start_time, end_time, exist_end_time = utils.time_to_seconds(modified_response[i]['time'])
            if not exist_end_time:
                end_time, _, _ = utils.time_to_seconds(modified_response[i + 1]['time'])
            prediction = modified_response[i]['text']
            if prediction not in scope:
                best_text, best_sim = self.reviser.get_similarity(prediction, scope)
                if best_sim > 0.9:
                    prediction = best_text
                else:
                    print(f"Invalid action prediction: {prediction}")
                    prediction = "none"
                    num_inconsistent_output += 1
            
            for t in range(start_time, end_time):
                predictions[t] = prediction

        last_start_time, last_end_time, exist_end_time = utils.time_to_seconds(modified_response[-1]['time'])
        if modified_response[-1]['text'] not in scope:
            best_text, best_sim = self.reviser.get_similarity(modified_response[-1]['text'], scope)
            if best_sim > 0.9:
                last_action = best_text
            else:
                print(f"Invalid action prediction: {modified_response[-1]['text']}")
                last_action = "none"
            num_inconsistent_output += 1
        else:
            last_action = modified_response[-1]['text']
        
        # print("Last action:", last_action)
        # print("Last start time:", last_start_time)
        # print("Last end time:", last_end_time)
        if not exist_end_time:
            for t in range(last_start_time, duration):
                predictions[t] = last_action
        else:
            for t in range(last_start_time, last_end_time):
                predictions[t] = last_action
        
        self.inference_time = time.time() - inference_start_time
        self.inference_cost = total_cost
        # print(f"Elapsed time: {self.inference_time:.2f} seconds")
        # print(f"Total cost: ${self.inference_cost:.2f}")
        if num_inconsistent_output > 0:
            print(f"Ratio of inconsistent outputs: {num_inconsistent_output}/{len(predictions)}")

        if fps != 1:
            actions_per_second = []
            for start_idx in range(0, len(predictions), fps):
                end_idx = min(start_idx + fps, len(predictions))  # prevent out of index error
                majority_action = utils.get_majority_vote(predictions[start_idx:end_idx])
                actions_per_second.append(majority_action)
            assert abs(len(actions_per_second) - original_duration) <= 3 # predicted duration should be close to the original duration
            predictions = actions_per_second[:original_duration]
            os.remove(video_path)

        output = pd.DataFrame({"seconds": range(len(predictions)), "predictions": predictions})
        if output.iloc[-1]["seconds"] != original_duration:
            # add as many rows as needed to match the duration of the video
            new_seconds, new_predictions = [], []
            for t in range(output.iloc[-1]["seconds"] + 1, original_duration + 1):
                new_seconds.append(t)
                new_predictions.append(output.iloc[-1]["predictions"])
            new_df = pd.DataFrame({"seconds": new_seconds, "predictions": new_predictions})
            output = pd.concat([output, new_df], ignore_index=True)
        
        utils.print_output(output)
        if show:
            utils.visualize_output(output["predictions"], save_path=None, legend_classes=scope, second_width=1, figsize=(15, 3))
                                   
        if save:
            output.to_csv(os.path.join(output_folder_dir, f"predictions_{self.model_name}.csv"), index=False)
        if os.path.exists("temp.mp4"):
            os.remove("temp.mp4")
        return output
    
    def classify_video(self, video_path, prompt="", scope=[], fps=16, temperature=0.0, max_tokens=256):
        
        video_file = self.agent.files.upload(file=video_path)
        # video_file = genai.upload_file(path=video_path) #NOTE: previous version

        # Check whether the file is ready to be used.
        while video_file.state.name == "PROCESSING":
            print('.', end='')
            time.sleep(1)
            video_file = self.agent.files.get(name=video_file.name)
            # video_file = genai.get_file(video_file.name) #NOTE: previous version

        if video_file.state.name == "FAILED":
            raise ValueError(video_file.state.name)

        ###############################################################
        # response = self.agent.generate_content( #NOTE: previous version
        #     contents=[video_file, GeminiAnalysisMode['ACT_DETECTION']],

        # )
        
        response = self.agent.models.generate_content(
            model=self.model_name,
            config=types.GenerateContentConfig(
                max_output_tokens=max_tokens,
                temperature=temperature),
            contents=[video_file, prompt]
        )
        ###############################################################
        prediction = response.candidates[0].content.parts[0].text
        prediction = utils.extract_text_from_tags(prediction, tag_name="task")
        
        if prediction not in scope:
            best_text, best_sim = self.reviser.get_similarity(prediction, scope)
            prediction = best_text
            print(f"Invalid prediction is changed: {prediction} -> {best_text}")
        
        return prediction
        
class AgentTwelveLabs():
    def __init__(self, logger=None, model_name="pegasus1.2", api_key=None, region=None):
        """
        Initialize TwelveLabs Agent class.
        Args:
            logger (Logger): Logger instance for logging information. NOTE: Unused.
            model_name (str): Name of the model to use. Defaults to "pegasus1.2".
            api_key (str): API key for authentication. Defaults to None.
            region (str, optional): Region for the API. Defaults to None. NOTE: Unused.
        """
        self.logger = logger
        self.model_name = model_name
        self.pricing_dict = api_pricing_dict[model_name]
        self.api_key = api_key
        self.region = region
        self.max_video_size = None #TODO: Set the maximum video size for the model
        self.agent = TwelveLabs(api_key=api_key)
    
    def check_video_file(self, video_path, index_obj):
        videos = self.agent.index.video.list(index_obj.id)
        video_files = [video.system_metadata.filename for video in videos]
        video_file = os.path.basename(video_path)
        if video_file not in video_files:
            task = self.agent.task.create(index_id=index_obj.id, file=video_path, language="en")
        for video in videos:
            if video.system_metadata.filename == video_file:
                return video
        return None

    def ask_about_video(self, video_path, prompt="", scope=[], fps=1, temperature=0.0, max_tokens=128, output_folder_dir="output", save=False, show=False):
        # TODO: fps adjustment
        duration = original_duration = int(utils.get_video_duration(video_path))

        duration = utils.get_video_duration(video_path)
        if duration is None:
            raise ValueError(f"Unable to get the duration of the video file: {video_path}")
        
        # video_path = utils.resize_video(video_path, "temp.mp4", self.max_video_size)
        total_cost = 0.0
        start_time = time.time()
        predictions = ["none"] * (duration+1)
        
        num_inconsistent_output = 0
        
        # For TwelveLabs, the video file should be uploaded to the server.
        # TODO: Handle the error when index_obj is None
        # TODO: Delete the uploaded video file after the inference is done.
        index_obj = self.agent.index.list()[0]

        video = self.check_video_file(video_path, index_obj)
        if video is None:
            raise ValueError("Failed to upload the video file to the server.")
        
        inference_start_time = time.time()
        response = self.agent.generate.text(video_id=video.id,
                               prompt=prompt,
                               temperature=temperature,
                               )
        predicted_actions = utils.extract_text_from_tags(response.data, tag_name="action")
        predicted_actions = re.findall(r'\(([^,]+), ([^)]+)\)', predicted_actions)
        # TODO: Enforce predicted_actions includes correct format of action and time
        for i in range(len(predicted_actions) - 1):
            start_time = utils.time_to_seconds(predicted_actions[i][1])
            end_time = utils.time_to_seconds(predicted_actions[i + 1][1])
            prediction = predicted_actions[i][0]
            if prediction not in scope:
                prediction = "none"
                num_inconsistent_output += 1
            
            for t in range(start_time, end_time):
                predictions[t] = prediction

        last_start_time = utils.time_to_seconds(predicted_actions[-1][1])
        for t in range(last_start_time, duration+1):
            predictions[t] = predicted_actions[-1][0]

        self.inference_time = time.time() - inference_start_time
        self.inference_cost = total_cost
        print(f"Elapsed time: {self.inference_time:.2f} seconds")
        # print(f"Total cost: ${self.inference_cost:.2f}")
        if num_inconsistent_output > 0:
            print(f"Ratio of inconsistent outputs: {num_inconsistent_output}/{len(predictions)}")
        
        # if fps != 1:
        #     actions_per_second = []
        #     for start_idx in range(0, len(predictions), fps):
        #         end_idx = min(start_idx + fps, len(predictions))  # prevent out of index error
        #         majority_action = utils.get_majority_vote(predictions[start_idx:end_idx])
        #         actions_per_second.append(majority_action)
        #     assert abs(len(actions_per_second) - original_duration) <= 3 # predicted duration should be close to the original duration
        #     predictions = actions_per_second[:original_duration]
        #     os.remove(video_path)

        output = pd.DataFrame({"seconds": range(len(predictions)), "predictions": predictions})
        if output.iloc[-1]["seconds"] != original_duration:
            # add as many rows as needed to match the duration of the video
            new_seconds, new_predictions = [], []
            for t in range(output.iloc[-1]["seconds"] + 1, original_duration + 1):
                new_seconds.append(t)
                new_predictions.append(output.iloc[-1]["predictions"])
            new_df = pd.DataFrame({"seconds": new_seconds, "predictions": new_predictions})
            output = pd.concat([output, new_df], ignore_index=True)
        
        utils.print_output(output)
        if show:
            utils.visualize_output(output["predictions"], save_path=None, legend_classes=scope, second_width=1, figsize=(15, 3))
                                   
        if save:
            output.to_csv(os.path.join(output_folder_dir, f"predictions_{self.model_name}.csv"), index=False)
        if os.path.exists("temp.mp4"):
            os.remove("temp.mp4")
        return output
    
class AgentBERT():
    def __init__(self):
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = BertModel.from_pretrained('bert-base-uncased').to(self.device)
    
    def get_embeddings(self, texts):
        inputs = self.tokenizer(texts, return_tensors='pt', padding=True, truncation=True).to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
            return outputs.last_hidden_state[:, 0, :]  # CLS token embedding
    
    def get_similarity(self, text1, text2):
        if isinstance(text1, str):
            text1 = [text1]
        if isinstance(text2, str):
            text2 = [text2]
        
        query_emb = self.get_embeddings(text1)          # shape: (1, hidden)
        list_embs = self.get_embeddings(text2)          # shape: (N, hidden)
        cos_sims = F.cosine_similarity(query_emb, list_embs)  # shape: (N,)
        best_sim, best_idx = torch.max(cos_sims, dim=0)
        best_text = text2[best_idx.item()]
        return best_text, best_sim.item()