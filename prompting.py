import os, json

from pydantic import BaseModel, Field
from typing import List, Literal, Optional

class Prompt:
    def __init__(self, action_classes):
        # self.action_classes = ["lift", "pick", "carry", "place", "hold", "pinch", "push", "pull", "install", "tighten", "remove", "cut", "lower", "turn", "open", "close", "none"]
        # self.action_classes = ["carry/hold", "close", "cut", "lift/pick up", "open", "press", "pull", "push", "put down", "turn", "none"]  # for AVA dataset
        # self.action_classes = ["lift", "pull", "hold", "place", "load", "remove", "install", "apply", "put", "insert", "tighten", "unload", "open", "close", "cut", "take", "none"]  # for COIN Dataset
        # self.action_classes = json.load(open("./COIN_videos/coin_action_conversion.json"))
        # self.action_classes = [key for key in self.action_classes.keys()]
        # self.action_classes.append("none")
        self.action_classes = action_classes
    
    def get_prompt(self, model_name, example_action):
        if model_name in ["gemini-2.0-flash-exp", "gemini-1.5-pro", "gemini-2.0-flash"]:
            prompt_for_gemini =  f"Call set_timecodes once using the following instructions: Identify a worker's all actions in the video. Action class must be in the following list: {self.action_classes}. Example: {example_action}" # Also, identify the start time of each action.
            return prompt_for_gemini
        
        elif model_name in ["pegasus-2.0-flash"]:
            action_prompt_for_pegasus = f"Identify a worker's all actions. Identify their start times. Action class must be in the following list: {self.action_classes}, or type 'none' if the action is not listed. Place your answer between <action> tag and </action>. Example: <action>{example_action}</action>"

            return action_prompt_for_pegasus
        
        else:
            # prompt = f"What is the worker's action? You must choose from the following options: {self.action_classes}, or type 'none' if the action is not listed. Example: {example_action}"
            # prompt = f"Select the action term that the worker is performing in the given video from the following action list: <action>{self.action_classes}</action>, or type 'none' if the action is not listed. Place your answer between <action> tag and </action>. Example: <action>{example_action}</action>"
            prompt = f"""
            # GOAL: Your task is to distinguish whether the given vidoe frames contain a specific action or an irrelevant one (i.e., none), and if it is a specific action, identify exactly what action it is.
            # RETURN FORMAT: Place your answer between <action> tag and </action> tag.
            # EXAMPLE: <action>{example_action}</action>
            # WARNING: If the action is not listed, type <action>none</action>.
            # ACTION LIST: {self.action_classes}
            """
            return prompt
        
    def get_video_level_prompt(self, scope, example):
        prompt = f"Given the frames, identify the most relevant task from the following list: {scope} \
                    # RETURN FORMAT: Place your answer between <task> tag and </task> tag. \
                    # EXAMPLE: <task>{example}</task>"
        return prompt
    
    def get_revision_prompt(self, model_name):
        if model_name in ["gemini-2.0-flash-exp", "gemini-1.5-pro", "gemini-2.0-flash"]:
            prompt = ""
        elif model_name in ["pegasus-2.0-flash"]:
            prompt = ""
        else:
            prompt = """
            You made a prediction for the following task. Now, you have a chance to revise your answer.
            If your prediction is semantically or ergonomically similar to the reference below, revise your answer to match it.
            Wrap your revised answer inside <action>...</action>.
            If your prediction is not semantically or ergonomically close to the reference, leave it unchanged.
            Reference answer: 
            """
        return prompt

class Task(BaseModel):
    task: Literal[
    "AssembleBed", "AssembleSofa", "AssembleCabinet", "AssembleDesktopPC",
    "ChangeBikeChain", "PolishCar", "FuelCar", "ChangeCarTire",
    "ChangeBatteryOfWatch", "ChangeBikeTires", "CleanLaptopKeyboard",
    "Graft", "ArcWeld", "DrillHole", "FixLaptopScreenScratches", "LoadGreaseGun", "PatchBikeInnerTube",
    "ReplaceCarWindow", "ReplaceBatteryOnKeyToCar", "ReplaceBatteryOnTVControl",
    "InstallAirConditioner", "InstallBicycleRack", "InstallCeilingFan", "InstallCeramicTile",
    "InstallLicensePlateFrame", "InstallShowerHead", "InstallClosestool", "InstallWoodFlooring", "InstallCurtain",
    "ChangeMobilePhoneBattery", "PumpUpBicycleTire", "ReplaceABulb", "ReplaceCarFuse", "ReplaceElectricalOutlet",
    "ReplaceFilterForAirPurifier", "ReplaceLaptopScreen", "ReplaceRefrigeratorWaterFilter", "ReplaceLightSocket",
    "ReplaceGraphicsCard", "ReplaceHardDisk", "ReplaceMemoryChip", "AssembleOfficeChair", "PractiseWeightLift"
    ] = Field(..., description="Given the frames, describe the most relevant task!")
    # COIN_task_interested = [
    #     "AssembleBed", "AssembleSofa", "AssembleCabinet", "AssembleDesktopPC",
    #     "ChangeBikeChain", "PolishCar", "FuelCar", "ChangeCarTire",
    #     "ChangeBatteryOfWatch", "ChangeBikeTires", "CleanLaptopKeyboard",
    #     "Graft", "ArcWeld", "DrillHole", "FixLaptopScreenScratches", "LoadGreaseGun", "PatchBikeInnerTube",
    #     "ReplaceCarWindow", "ReplaceBatteryOnKeyToCar", "ReplaceBatteryOnTVControl",
    #     "InstallAirConditioner", "InstallBicycleRack", "InstallCeilingFan", "InstallCeramicTile",
    #     "InstallLicensePlateFrame", "InstallShowerHead", "InstallClosestool", "InstallWoodFlooring", "InstallCurtain",
    #     "ChangeMobilePhoneBattery", "PumpUpBicycleTire", "ReplaceABulb", "ReplaceCarFuse", "ReplaceElectricalOutlet",
    #     "ReplaceFilterForAirPurifier", "ReplaceLaptopScreen", "ReplaceRefrigeratorWaterFilter", "ReplaceLightSocket",
    #     "ReplaceGraphicsCard", "ReplaceHardDisk", "ReplaceMemoryChip", "AssembleOfficeChair", "PractiseWeightLift"
    # ]

# GeminiAnalysisMode = {
#                     "AV_CAPTIONS": "For each scene in this video, generate captions that "
#                     "describe the scene along with any spoken text placed in quotation marks. "
#                     "Place each caption into an object sent to set_timecodes with the timecode of the caption in the video.",

#                     "PARAGRAPH": "Generate a paragraph that summarizes this video. Keep it to 3 to 5 "
#                     "sentences. Place each sentence of the summary into an object sent to set_timecodes with the "
#                     "timecode of the sentence in the video.",

#                     "KEY_MOMENTS": "Generate bullet points for the video. Place each bullet point into an "
#                         "object sent to set_timecodes with the timecode of the bullet point in the video.",

#                     "TABLE": "Choose 5 key shots from this video and call set_timecodes_with_objects with the "
#                     "timecode, text description of 10 words or less, and a list of objects visible in the scene "
#                     "(with representative emojis).",

#                     "HAIKU": "Generate a haiku for the video. Place each line of the haiku into an object sent "
#                     "to set_timecodes with the timecode of the line in the video. Make sure to follow the syllable "
#                     "count rules (5-7-5).",

#                     "CHART": "Generate chart data for this video based on the following instructions: \n"
#                     "count the number of people. Call set_timecodes_with_numeric_values once with the list "
#                     "of data values and timecodes.",

#                     "CUSTOM": "Call set_timecodes once using the following instructions: ",

#                     "AV_DESCRIPTIONS": "For each scene in this video, generate spoken text and descriptions that "
#                     "describe the scene along with any spoken text placed in quotation marks. "
#                     "Place each section into an object sent to set_timecodes_with_descriptions with the timecode of the caption in the video, the spoken text and the visual description.",

#                     "ACT_DETECTION":action_prompt_for_gemini,
#                 }