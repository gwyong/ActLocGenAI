import os, glob, json, time, random, sys
import pandas as pd
import genais, utils, prompting
from tqdm import tqdm

seed = 42
random.seed(seed)
model_name = "gpt-4o"
# model_name = "gemini-2.0-flash"
# model_name = "gemini-2.5-pro-preview-03-25--image"
# model_name = "gemini-2.5-pro-exp-03-25--image"
# model_name = "claude-3-7-sonnet-latest"
fps = 4
max_tokens = 2048 # 2048
save_metadata_df = False
output_folder_dir = "output"

df_task_step = pd.read_csv("./COIN_videos/filtered_COIN_videos/task_step.csv")
# action_classes = [" - ".join([df_task_step.iloc[i]["Task"], df_task_step.iloc[i]["Step"]]) for i in range(len(COIN_task_interested))]
action_classes = df_task_step["unified_3"].unique().tolist()
act_classes, obj_classes = [], []
for action in action_classes:
    if action == "none":
        act_classes.append("none")
        obj_classes.append("none")
    else:
        tokens = action.split(" ")
        obj_classes.append(tokens[-1])
        act_classes.append(" ".join(tokens[:-1]))
action_classes = list(set(act_classes))
object_classes = list(set(obj_classes))
# prompter = prompting.Prompt(action_classes)
prompter = prompting.Prompt(action_classes=action_classes, object_classes=object_classes)
example_action = random.choice(action_classes)
# print(f"# of Total Actions: {len(action_classes)}| Example Action: {example_action}")
print(f"# of Total Actions: {len(action_classes)}| Example Action: {action_classes[0]}")
print(f"# of Total Objects: {len(object_classes)}| Example Object: {object_classes[0]}")

if "none" not in action_classes:
    action_classes.append("none")
query = prompter.get_prompt(model_name, example_action=example_action)
print(f"Query: {query}")

# dataset_folder_path = "./ava/train_preprocessed"
dataset_folder_path = "./COIN_videos/filtered_COIN_videos"
json_path = os.path.join(output_folder_dir, f"predictions_{model_name}_{fps}.json")

api_key_path = "APIKEY/api_key.json"
api_key_dict = utils.load_json(api_key_path)

if model_name in ["gpt-4o", "gpt-4o-mini", "o1"]:
    agent = genais.AgentOpenAI(model_name=model_name, api_key=api_key_dict["OpenAI_yong"])
    # agent = genais.AgentOpenAI(model_name=model_name, api_key=api_key_dict["OpenAI_UM"])
elif model_name in ["claude-3-5-sonnet-latest", "claude-3-5-sonnet", "claude-3-7-sonnet-latest"]:
    agent = genais.AgentAnthropic(api_key=api_key_dict["Anthropic_yong"])
elif model_name in ["gemini-2.0-flash-exp", "gemini-1.5-pro", "gemini-2.0-flash", "gemini-2.5.pro-exp-03-25--image", "gemini-2.5-pro-preview-03-25--image", "gemini-2.5-pro-exp-03-25--image"]:
    agent = genais.AgentGoogle(model_name=model_name, api_key=api_key_dict["Gemini_yong"])

num_testing = 10
video_files = glob.glob(os.path.join(dataset_folder_path, "*.mp4"))
selected_video_files = random.sample(video_files, num_testing)

start_time = time.time()
for video_path in tqdm(selected_video_files):
# for video_path in tqdm(video_files):
    
    if os.path.exists(json_path):
        with open(json_path, "r") as f:
            data = json.load(f)
            if video_path in data:
                continue
    
    df_predictions = agent.ask_about_video(video_path, prompt=query, scope=action_classes, fps=fps, temperature=0.0, max_tokens=max_tokens, output_folder_dir=output_folder_dir, save=save_metadata_df)
    utils.append_df_to_json(df_predictions, video_path, json_path)

    time.sleep(1)
    break

end_time = time.time()
print(f"Total Time taken: {end_time - start_time} seconds")
