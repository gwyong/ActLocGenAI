import os, glob, json
import cv2
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

from tqdm import tqdm

def plot_action_bars(actual_actions, predicted_actions, save_path=None, color_map=None, legend_classes=None, second_width=1, figsize=(15, 3)):
    assert len(actual_actions) == len(predicted_actions), "The number of actual and predicted actions must be the same."
    num_seconds = len(actual_actions)

    # time_ticks = np.arange(0, num_seconds * second_width + 1, second_width)
    time_ticks = range(0, num_seconds * second_width, second_width)
    fig, ax = plt.subplots(2, 1, figsize=figsize, sharex=True)

    # upper bar: actual actions
    ax[0].bar(
        # time_ticks[:-1],
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
        # time_ticks[:-1],
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

def testing_ava(path_gt_second_action_dict, path_pred_second_action_dict, output_folder_path="./output", save=True, save_fig=False):
    # TODO: Someday, we will update this function.

    ava_action_conversion = {
        "carry/hold": "carry/hold (an object)",
        "close": "close (e.g., a door, a box)",
        "cut": "cut",
        "lift/pick up":"lift/pick up",
        "open":"open (e.g., a window, a car door)",
        "press":"press",
        "pull":"pull (an object)",
        "push":"push (an object)",
        "put down":"put down",
        "turn": "turn (e.g., a screwdriver)",
        "none": "none",
    }
    ava_actions = list(set(list(ava_action_conversion.values())))
    color_map = {
        action: ("gray" if action == "none" else plt.cm.tab20(i / len(ava_actions)))
        for i, action in enumerate(ava_actions)
    }

    with open(path_gt_second_action_dict, "r") as f:
        gt_second_action_dicts = json.load(f)
    with open(path_pred_second_action_dict, "r") as f:
        pred_second_action_dicts = json.load(f)

    df_result = pd.DataFrame(columns=["video_path", "total_acc"])
    video_paths, total_accs = [], []
    for video_path in tqdm(pred_second_action_dicts.keys()):
        gt_second_action_dict = gt_second_action_dicts[video_path]
        pred_second_action_dict = pred_second_action_dicts[video_path]

        if abs(len(gt_second_action_dict) - len(pred_second_action_dict)) > 1:
            print(f"The number of seconds in the ground-truth and prediction files are quite different. | video_path: {video_path}")

        sorted_gt_seconds = sorted(gt_second_action_dict.keys(), key=int)
        sorted_pred_seconds = sorted(pred_second_action_dict.keys(), key=int)

        gt_actions = []
        pred_actions = []
        total_secs = len(sorted_gt_seconds)
        correct_secs = 0

        for i in range(len(sorted_gt_seconds)):
            gt_second = sorted_gt_seconds[i]
            pred_second = sorted_pred_seconds[i]

            pred_action = pred_second_action_dict[pred_second]
            pred_action = ava_action_conversion[pred_action]
            pred_actions.append(pred_action)

            candidate_gt_actions = gt_second_action_dict[gt_second]
            if pred_action in candidate_gt_actions:
                gt_action = pred_action
                correct_secs += 1
            
            else:
                if candidate_gt_actions == ["none"]:
                    gt_action = "none"
                else:
                    # delete "none" from candidate_gt_actions
                    candidate_gt_actions.remove("none")
                    gt_action = candidate_gt_actions[0]
            gt_actions.append(gt_action)
        
        total_acc = correct_secs / total_secs
        video_paths.append(video_path)
        total_accs.append(total_acc)

        if save_fig:
            save_path = os.path.join(output_folder_path, os.path.basename(video_path).replace(".mp4", ".png"))
            plot_action_bars(gt_actions, pred_actions, save_path=save_path, color_map=color_map, legend_classes=ava_actions)
    
    df_result["video_path"] = video_paths
    df_result["total_acc"] = total_accs
    print(f"Average Total Accuracy: {df_result['total_acc'].mean()}")

    if save:
        model_name = path_pred_second_action_dict.split("/")[-1].replace("predictions_", "").replace(".json", "")
        save_path = os.path.join(output_folder_path, f"ava_results_{model_name}.csv")
        df_result.to_csv(save_path, index=False)

def testing_coin(path_gt_second_action_dict, path_pred_second_action_dict, output_folder_path="./output", save=True, save_fig=False):
    # TODO: Someday, we will update this function.

    dataset_action_conversion = {"install legs of sofa": "install",
                              "put on sofa cover": "put",
                              "place cushion and backrest": "place",
                              "lift the barbell directly over the head": "lift",
                              "stand up with straight legs": "none",
                              "pull the barbell to the chin": "pull",
                              "lift the barbell from the chin over the head": "lift",
                              "place license plate cover": "place",
                              "screw and fix the cover": "none",
                              "place license plate": "place",
                              "assemble the frame": "none",
                              "install vertical boards": "install",
                              "install horizontal boards": "install",
                              "place the board on each side": "place",
                              "put all bed boxes together": "put",
                              "place the bed board": "place",
                              "tile the wall": "none",
                              "wipe adhesive and bedding layer": "none",
                              "load the inner tube": "load",
                              "load the tire": "load",
                              "remove the light socket and disconnect the wire": "remove",
                              "install the light socket": "install",
                              "install the bulb and light housing or shell": "install",
                              "unscrew the screw": "none",
                              "jack up the car": "none",
                              "squat and hold the barbell": "hold",
                              "put down the barbell": "put",
                              "cut the old chain": "cut",
                              "take old chain away": "take",
                              "install head of the bed": "install",
                              "place board at the bottom": "place",
                              "place mattress on the bed": "place",
                              "install the wheels for the base": "install",
                              "assemble the cushion and the backrest": "none",
                              "connect the chair and the base": "none",
                              "take out the old filter": "take",
                              "remove the cap of the new filter": "remove",
                              "install the new filter": "install",
                              "install armrest on sofa": "install",
                              "install the new bulb": "install",
                              "install the light shell/housing/support": "install",
                              "remove the tire": "remove",
                              "pump up the tire": "none",
                              "fit on the boards": "none",
                              "cut the raw boards": "cut",
                              "put on the tire": "put",
                              "tighten screws": "tighten",
                              "remove the inner tube": "remove",
                              "put every parts mentioned together": "put",
                              "install the sofa back": "install",
                              "install the new chain": "install",
                              "take out the shell": "take",
                              "take out the filter": "take",
                              "fit on the filter": "none",
                              "fit on the shell": "none",
                              "remove inner tube": "remove",
                              "put inner tube back": "put",
                              "clean the scratch": "none",
                              "apply polishing paste on the surface": "apply",
                              "remove the light housing or shell and bulb": "remove",
                              "install legs on the bed": "install",
                              "mount the bracket to the back of the car": "none",
                              "place the bicycle on the bracket": "place",
                              "fasten the strap": "none",
                              "knock in the nails": "none",
                              "prepare adhesive": "none",
                              "check ground and layout planning": "none",
                              "fill the gap": "none",
                              "prepare the ground floor": "none",
                              "unload the wheel": "unload",
                              "load the wheel": "load",
                              "open the fuel tank cap": "open",
                              "insert oil gun in the car": "insert",
                              "operate the control panel": "none",
                              "pullthe  oil gun out": "pull",
                              "close the fuel tank cap": "close",
                              "connect the wire again": "none",
                              "look for leaks": "none",
                              "use sandpaper/metal to polish rubber near leak": "none",
                              "apply glue": "apply",
                              "paste patch": "none",
                              "install window panel": "install",
                              "connect air conditioners and windows": "none",
                              "polish with polisher": "none",
                              "take out the battery": "take",
                              "unscrew the screws used to fix the screen": "none",
                              "pull out the screen connector and remove the screen": "pull",
                              "wrap the pipe band": "none",
                              "install the new shower head": "install",
                              "apply some glue on the borads": "apply",
                              "fix and fasten the upper and lower ropes to car": "none",
                              "open the laptop rear cover": "open",
                              "remove the old hard disk": "remove",
                              "install stand of the seat": "install",
                              "install the air nozzle": "install",
                              "pump up to the tire": "none",
                              "remove the air nozzle": "remove",
                              "place crossbar on shelves": "place",
                              "install two shelves": "install",
                              "hang up curtains": "none",
                              "open cover": "open",
                              "remove battery": "remove",
                              "open the car key cover": "open",
                              "take out the car key battery": "take",
                              "put in the battery": "put",
                              "close the car key cover": "close",
                              "draw lines to mark the hole": "none",
                              "drill with an electric drill": "none",
                              "install fan tray": "install",
                              "install fans and lights": "install",
                              "remove surface of the door": "remove",
                              "remove old glass from window lift": "remove",
                              "cut branches": "cut",
                              "connect branches": "none",
                              "glue on window frame": "none",
                              "install new glass on window lift": "install",
                              "tighten the valve and screw on the valve cap": "tighten",
                              "fix window board": "none",
                              "pull out a lock lever at bottom of gun": "pull",
                              "put in a new grease container": "put",
                              "open the lid of new container": "open",
                              "screw gun head back": "none",
                              "try to press gun head, spray residual old grease": "none",
                              "replace the old shower head": "none",
                              "install the door": "install",
                              "screw off the valve cap and open the valve": "none",
                              "hold electrode with a welding tong": "hold",
                              "wear a eye protection helmet": "none",
                              "weld along the weld line": "none",
                              "clean weld line with a hammer": "none",
                              "open up the cover": "open",
                              "take out the fuse": "take",
                              "fit on the fuse": "none",
                              "clean the surface": "none",
                              "wind the junction to fasten the connection": "none",
                              "install fan bracket": "install",
                              "pick up the back cover of the phone with the cymbal": "none",
                              "remove the components of the fixed battery": "remove",
                              "take down the old battery": "take",
                              "close up the cover": "close",
                              "cut tiles": "cut",
                              "remove the old wax ring and clean": "remove",
                              "install the new wax ring": "install",
                              "find the position of the hole": "none",
                              "apply toothpaste or other cleaning agent to scratch": "apply",
                              "wipe the scratch with cloth": "none",
                              "wipe the toothpaste": "none",
                              "install the closestool again": "install",
                              "remove the light shell/housing/support": "remove",
                              "take out the old bulb": "take",
                              "install fan frame": "install",
                              "connect the water pipe": "none",
                              "none": "none"}
    dataset_actions = list(set(list(dataset_action_conversion.values())))
    color_map = {
        action: ("gray" if action == "none" else plt.cm.tab20(i / len(dataset_actions)))
        for i, action in enumerate(dataset_actions)
    }

    with open(path_gt_second_action_dict, "r") as f:
        gt_second_action_dicts = json.load(f)
    with open(path_pred_second_action_dict, "r") as f:
        pred_second_action_dicts = json.load(f)

    df_result = pd.DataFrame(columns=["video_path", "total_acc", "total_acc_wo_none"])
    video_paths, total_accs, total_accs_wo_none = [], [], []
    for video_path in tqdm(pred_second_action_dicts.keys()):
        gt_second_action_dict = gt_second_action_dicts[video_path]
        pred_second_action_dict = pred_second_action_dicts[video_path]

        if abs(len(gt_second_action_dict) - len(pred_second_action_dict)) > 1:
            print(f"The number of seconds in the ground-truth and prediction files are quite different. | video_path: {video_path}")

        sorted_gt_seconds = sorted(gt_second_action_dict.keys(), key=int)
        sorted_pred_seconds = sorted(pred_second_action_dict.keys(), key=int)
        
        gt_actions = []
        pred_actions = []
        total_secs = len(sorted_gt_seconds)
        target_secs = 0
        correct_secs_none = 0
        correct_secs = 0

        for i in range(len(sorted_gt_seconds)):
            gt_second = sorted_gt_seconds[i]
            if gt_second not in pred_second_action_dict:
                continue
            pred_second = sorted_pred_seconds[i]

            pred_action = pred_second_action_dict[pred_second]
            if pred_action not in dataset_actions:
                if pred_action.split(" ")[0] in dataset_actions:
                    pred_action = pred_action.split(" ")[0]
                else:
                    pred_action = "none"
            pred_actions.append(pred_action)
            gt_action = gt_second_action_dict[gt_second]
            gt_action = dataset_action_conversion[gt_action]
            gt_actions.append(gt_action)
            
            if gt_action == "none":
                if pred_action == gt_action:
                    correct_secs_none += 1
            else:
                if pred_action == gt_action:
                    correct_secs += 1
                target_secs += 1
        
        total_acc = (correct_secs+correct_secs_none) / total_secs
        if target_secs == 0:
            total_acc_wo_none = 0
        else:
            total_acc_wo_none = correct_secs / target_secs
        video_paths.append(video_path)
        total_accs.append(total_acc)
        total_accs_wo_none.append(total_acc_wo_none)

        if save_fig:
            save_path = os.path.join(output_folder_path, os.path.basename(path_pred_second_action_dict).replace(".json", "_")+os.path.basename(video_path).replace(".mp4", ".png"))
            plot_action_bars(gt_actions, pred_actions, save_path=save_path, color_map=color_map, legend_classes=dataset_actions)
    
    df_result["video_path"] = video_paths
    df_result["total_acc"] = total_accs
    df_result["total_acc_wo_none"] = total_accs_wo_none
    print(f"Average Total Accuracy: {df_result['total_acc'].mean()}")
    print(f"Average Total Accuracy (w/o 'none'): {df_result['total_acc_wo_none'].mean()}")

    if save:
        model_name = path_pred_second_action_dict.split("/")[-1].replace("predictions_", "").replace(".json", "")
        save_path = os.path.join(output_folder_path, f"coin_results_{model_name}.csv")
        df_result.to_csv(save_path, index=False)

def testing_coin_v2(path_gt_second_action_dict, path_pred_second_action_dict, output_folder_path="./output", save=True, save_fig=False):
    # TODO: Someday, we will update this function.

    # dataset_action_conversion = {"install legs of sofa": "install",
    #                           "put on sofa cover": "put",
    #                           "place cushion and backrest": "place",
    #                           "lift the barbell directly over the head": "lift",
    #                           "stand up with straight legs": "none",
    #                           "pull the barbell to the chin": "pull",
    #                           "lift the barbell from the chin over the head": "lift",
    #                           "place license plate cover": "place",
    #                           "screw and fix the cover": "none",
    #                           "place license plate": "place",
    #                           "assemble the frame": "none",
    #                           "install vertical boards": "install",
    #                           "install horizontal boards": "install",
    #                           "place the board on each side": "place",
    #                           "put all bed boxes together": "put",
    #                           "place the bed board": "place",
    #                           "tile the wall": "none",
    #                           "wipe adhesive and bedding layer": "none",
    #                           "load the inner tube": "load",
    #                           "load the tire": "load",
    #                           "remove the light socket and disconnect the wire": "remove",
    #                           "install the light socket": "install",
    #                           "install the bulb and light housing or shell": "install",
    #                           "unscrew the screw": "none",
    #                           "jack up the car": "none",
    #                           "squat and hold the barbell": "hold",
    #                           "put down the barbell": "put",
    #                           "cut the old chain": "cut",
    #                           "take old chain away": "take",
    #                           "install head of the bed": "install",
    #                           "place board at the bottom": "place",
    #                           "place mattress on the bed": "place",
    #                           "install the wheels for the base": "install",
    #                           "assemble the cushion and the backrest": "none",
    #                           "connect the chair and the base": "none",
    #                           "take out the old filter": "take",
    #                           "remove the cap of the new filter": "remove",
    #                           "install the new filter": "install",
    #                           "install armrest on sofa": "install",
    #                           "install the new bulb": "install",
    #                           "install the light shell/housing/support": "install",
    #                           "remove the tire": "remove",
    #                           "pump up the tire": "none",
    #                           "fit on the boards": "none",
    #                           "cut the raw boards": "cut",
    #                           "put on the tire": "put",
    #                           "tighten screws": "tighten",
    #                           "remove the inner tube": "remove",
    #                           "put every parts mentioned together": "put",
    #                           "install the sofa back": "install",
    #                           "install the new chain": "install",
    #                           "take out the shell": "take",
    #                           "take out the filter": "take",
    #                           "fit on the filter": "none",
    #                           "fit on the shell": "none",
    #                           "remove inner tube": "remove",
    #                           "put inner tube back": "put",
    #                           "clean the scratch": "none",
    #                           "apply polishing paste on the surface": "apply",
    #                           "remove the light housing or shell and bulb": "remove",
    #                           "install legs on the bed": "install",
    #                           "mount the bracket to the back of the car": "none",
    #                           "place the bicycle on the bracket": "place",
    #                           "fasten the strap": "none",
    #                           "knock in the nails": "none",
    #                           "prepare adhesive": "none",
    #                           "check ground and layout planning": "none",
    #                           "fill the gap": "none",
    #                           "prepare the ground floor": "none",
    #                           "unload the wheel": "unload",
    #                           "load the wheel": "load",
    #                           "open the fuel tank cap": "open",
    #                           "insert oil gun in the car": "insert",
    #                           "operate the control panel": "none",
    #                           "pullthe  oil gun out": "pull",
    #                           "close the fuel tank cap": "close",
    #                           "connect the wire again": "none",
    #                           "look for leaks": "none",
    #                           "use sandpaper/metal to polish rubber near leak": "none",
    #                           "apply glue": "apply",
    #                           "paste patch": "none",
    #                           "install window panel": "install",
    #                           "connect air conditioners and windows": "none",
    #                           "polish with polisher": "none",
    #                           "take out the battery": "take",
    #                           "unscrew the screws used to fix the screen": "none",
    #                           "pull out the screen connector and remove the screen": "pull",
    #                           "wrap the pipe band": "none",
    #                           "install the new shower head": "install",
    #                           "apply some glue on the borads": "apply",
    #                           "fix and fasten the upper and lower ropes to car": "none",
    #                           "open the laptop rear cover": "open",
    #                           "remove the old hard disk": "remove",
    #                           "install stand of the seat": "install",
    #                           "install the air nozzle": "install",
    #                           "pump up to the tire": "none",
    #                           "remove the air nozzle": "remove",
    #                           "place crossbar on shelves": "place",
    #                           "install two shelves": "install",
    #                           "hang up curtains": "none",
    #                           "open cover": "open",
    #                           "remove battery": "remove",
    #                           "open the car key cover": "open",
    #                           "take out the car key battery": "take",
    #                           "put in the battery": "put",
    #                           "close the car key cover": "close",
    #                           "draw lines to mark the hole": "none",
    #                           "drill with an electric drill": "none",
    #                           "install fan tray": "install",
    #                           "install fans and lights": "install",
    #                           "remove surface of the door": "remove",
    #                           "remove old glass from window lift": "remove",
    #                           "cut branches": "cut",
    #                           "connect branches": "none",
    #                           "glue on window frame": "none",
    #                           "install new glass on window lift": "install",
    #                           "tighten the valve and screw on the valve cap": "tighten",
    #                           "fix window board": "none",
    #                           "pull out a lock lever at bottom of gun": "pull",
    #                           "put in a new grease container": "put",
    #                           "open the lid of new container": "open",
    #                           "screw gun head back": "none",
    #                           "try to press gun head, spray residual old grease": "none",
    #                           "replace the old shower head": "none",
    #                           "install the door": "install",
    #                           "screw off the valve cap and open the valve": "none",
    #                           "hold electrode with a welding tong": "hold",
    #                           "wear a eye protection helmet": "none",
    #                           "weld along the weld line": "none",
    #                           "clean weld line with a hammer": "none",
    #                           "open up the cover": "open",
    #                           "take out the fuse": "take",
    #                           "fit on the fuse": "none",
    #                           "clean the surface": "none",
    #                           "wind the junction to fasten the connection": "none",
    #                           "install fan bracket": "install",
    #                           "pick up the back cover of the phone with the cymbal": "none",
    #                           "remove the components of the fixed battery": "remove",
    #                           "take down the old battery": "take",
    #                           "close up the cover": "close",
    #                           "cut tiles": "cut",
    #                           "remove the old wax ring and clean": "remove",
    #                           "install the new wax ring": "install",
    #                           "find the position of the hole": "none",
    #                           "apply toothpaste or other cleaning agent to scratch": "apply",
    #                           "wipe the scratch with cloth": "none",
    #                           "wipe the toothpaste": "none",
    #                           "install the closestool again": "install",
    #                           "remove the light shell/housing/support": "remove",
    #                           "take out the old bulb": "take",
    #                           "install fan frame": "install",
    #                           "connect the water pipe": "none",
    #                           "none": "none"}
    # dataset_actions = list(dataset_action_conversion.keys())
    dataset_actions = pd.read_csv("./COIN_videos/filtered_COIN_videos/task_step.csv")["Step"].unique().tolist()
    dataset_actions.append("none")
    
    color_map = {
        action: ("gray" if action == "none" else plt.cm.tab20(i / len(dataset_actions)))
        for i, action in enumerate(dataset_actions)
    }

    with open(path_gt_second_action_dict, "r") as f:
        gt_second_action_dicts = json.load(f)
    with open(path_pred_second_action_dict, "r") as f:
        pred_second_action_dicts = json.load(f)

    df_result = pd.DataFrame(columns=["video_path", "total_acc", "total_acc_wo_none"])
    video_paths, total_accs, total_accs_wo_none = [], [], []
    
    for video_path in tqdm(pred_second_action_dicts.keys()):
        gt_second_action_dict = gt_second_action_dicts[video_path]
        pred_second_action_dict = pred_second_action_dicts[video_path]

        if abs(len(gt_second_action_dict) - len(pred_second_action_dict)) > 1:
            print(f"The number of seconds in the ground-truth and prediction files are quite different. | video_path: {video_path}")

        sorted_gt_seconds = sorted(gt_second_action_dict.keys(), key=int)
        sorted_pred_seconds = sorted(pred_second_action_dict.keys(), key=int)
        
        gt_actions = []
        pred_actions = []
        total_secs = len(sorted_gt_seconds)
        target_secs = 0
        correct_secs_none = 0
        correct_secs = 0

        for i in range(len(sorted_gt_seconds)):
            gt_second = sorted_gt_seconds[i]
            if gt_second not in pred_second_action_dict:
                continue
            pred_second = sorted_pred_seconds[i]

            pred_action = pred_second_action_dict[pred_second]
            if pred_action != "none":
                pred_action = pred_action.split(" - ")[-1]
                if pred_action not in dataset_actions:
                    print(f"Unknown action: {pred_action}")
            pred_actions.append(pred_action)
            
            gt_action = gt_second_action_dict[gt_second]
            gt_actions.append(gt_action)
            
            if gt_action == "none":
                if pred_action == gt_action:
                    correct_secs_none += 1
            else:
                if pred_action == gt_action:
                    correct_secs += 1
                target_secs += 1
        
        total_acc = (correct_secs+correct_secs_none) / total_secs
        if target_secs == 0:
            total_acc_wo_none = 0
        else:
            total_acc_wo_none = correct_secs / target_secs
        video_paths.append(video_path)
        total_accs.append(total_acc)
        total_accs_wo_none.append(total_acc_wo_none)

        if save_fig:
            save_path = os.path.join(output_folder_path, os.path.basename(path_pred_second_action_dict).replace(".json", "_")+os.path.basename(video_path).replace(".mp4", ".png"))
            plot_action_bars(gt_actions, pred_actions, save_path=save_path, color_map=color_map, legend_classes=dataset_actions)
    
    df_result["video_path"] = video_paths
    df_result["total_acc"] = total_accs
    df_result["total_acc_wo_none"] = total_accs_wo_none
    print(f"Average Total Accuracy: {df_result['total_acc'].mean()}")
    print(f"Average Total Accuracy (w/o 'none'): {df_result['total_acc_wo_none'].mean()}")

    if save:
        model_name = path_pred_second_action_dict.split("/")[-1].replace("predictions_", "").replace(".json", "")
        save_path = os.path.join(output_folder_path, f"coin_results_{model_name}.csv")
        df_result.to_csv(save_path, index=False)

def testing_coin_v3(path_gt_second_action_dict, path_pred_second_action_dict, output_folder_path="./output", save=True, save_fig=False):
    # TODO: Someday, we will update this function.

    df_dataset_actions = pd.read_csv("./COIN_videos/filtered_COIN_videos/task_step.csv")
    dataset_actions = df_dataset_actions["unified_2"].unique().tolist()
    dataset_action_conversion = {}
    for i in range(len(df_dataset_actions)):
        key = "---".join([df_dataset_actions.iloc[i]["Task"], df_dataset_actions.iloc[i]["Step"]])
        value = df_dataset_actions.iloc[i]["unified_2"]
        dataset_action_conversion[key] = value

    color_map = {
        action: ("gray" if action == "none" else plt.cm.tab20(i / len(dataset_actions)))
        for i, action in enumerate(dataset_actions)
    }

    with open(path_gt_second_action_dict, "r") as f:
        gt_second_action_dicts = json.load(f)
    with open(path_pred_second_action_dict, "r") as f:
        pred_second_action_dicts = json.load(f)

    df_result = pd.DataFrame(columns=["video_path", "total_acc", "total_acc_wo_none"])
    video_paths, total_accs, total_accs_wo_none = [], [], []
    
    for video_path in tqdm(pred_second_action_dicts.keys()):
        gt_second_action_dict = gt_second_action_dicts[video_path]
        pred_second_action_dict = pred_second_action_dicts[video_path]

        if abs(len(gt_second_action_dict) - len(pred_second_action_dict)) > 1:
            print(f"The number of seconds in the ground-truth and prediction files are quite different. | video_path: {video_path}")

        sorted_gt_seconds = sorted(gt_second_action_dict.keys(), key=int)
        sorted_pred_seconds = sorted(pred_second_action_dict.keys(), key=int)
        
        gt_actions = []
        pred_actions = []
        total_secs = len(sorted_gt_seconds)
        target_secs = 0
        correct_secs_none = 0
        correct_secs = 0

        for i in range(len(sorted_gt_seconds)):
            gt_second = sorted_gt_seconds[i]
            if gt_second not in pred_second_action_dict:
                continue
            pred_second = sorted_pred_seconds[i]

            pred_action = pred_second_action_dict[pred_second]
            if pred_action != "none":
                # pred_action = pred_action.split(" - ")[-1]
                if pred_action not in dataset_actions:
                    print(f"Unknown action: {pred_action}")
            pred_actions.append(pred_action)
            
            gt_action = gt_second_action_dict[gt_second]
            if gt_action != "none":
                video_name = os.path.basename(video_path).replace(".mp4", "")
                key = video_name.split("_")[0] + "---" + gt_action
                gt_action = dataset_action_conversion[key]
            gt_actions.append(gt_action)
            
            if gt_action == "none":
                if pred_action == gt_action:
                    correct_secs_none += 1
            else:
                if pred_action == gt_action:
                    correct_secs += 1
                target_secs += 1
        
        total_acc = (correct_secs+correct_secs_none) / total_secs
        if target_secs == 0:
            total_acc_wo_none = 0
        else:
            total_acc_wo_none = correct_secs / target_secs
        video_paths.append(video_path)
        total_accs.append(total_acc)
        total_accs_wo_none.append(total_acc_wo_none)

        if save_fig:
            save_path = os.path.join(output_folder_path, os.path.basename(path_pred_second_action_dict).replace(".json", "_")+os.path.basename(video_path).replace(".mp4", ".png"))
            plot_action_bars(gt_actions, pred_actions, save_path=save_path, color_map=color_map, legend_classes=dataset_actions)
    
    df_result["video_path"] = video_paths
    df_result["total_acc"] = total_accs
    df_result["total_acc_wo_none"] = total_accs_wo_none
    print(f"Average Total Accuracy: {df_result['total_acc'].mean()}")
    print(f"Average Total Accuracy (w/o 'none'): {df_result['total_acc_wo_none'].mean()}")

    if save:
        model_name = path_pred_second_action_dict.split("/")[-1].replace("predictions_", "").replace(".json", "")
        save_path = os.path.join(output_folder_path, f"coin_results_{model_name}.csv")
        df_result.to_csv(save_path, index=False)

def testing_coin_v4(path_gt_second_action_dict, path_pred_second_action_dict, output_folder_path="./output", save=True, save_fig=False):
    # TODO: Someday, we will update this function.
    # NOTE: For Gunwoo's Annotation

    df_dataset_actions = pd.read_csv("./COIN_videos/filtered_COIN_videos/task_step.csv")
    dataset_actions = df_dataset_actions["unified_2"].unique().tolist()
    dataset_action_conversion = {}
    for i in range(len(df_dataset_actions)):
        key = "---".join([df_dataset_actions.iloc[i]["Task"], df_dataset_actions.iloc[i]["Step"]])
        value = df_dataset_actions.iloc[i]["unified_2"]
        dataset_action_conversion[key] = value

    df_manual_annotation = pd.read_csv("./output/COIN_manual_annotation/GunwooAnnotation.csv")
    # drop the rows with missing values in the "annotation_1" column
    df_manual_annotation = df_manual_annotation.dropna(subset=["annotation_1"])
    # get the unique values in the "annotation_2" column
    new_dataset_actions = df_manual_annotation["annotation_2"].unique().tolist()
    # merge the two lists and remove duplicates
    merged_actions = list(set(dataset_actions + new_dataset_actions))
    if "drill object" in merged_actions:
        merged_actions.remove("drill object")
    if "hammar object" in merged_actions:
        merged_actions.remove("hammar object")
    dataset_actions = merged_actions

    color_map = {
        action: ("gray" if action == "none" else plt.cm.tab20(i / len(dataset_actions)))
        for i, action in enumerate(dataset_actions)
    }

    with open(path_gt_second_action_dict, "r") as f:
        gt_second_action_dicts = json.load(f)
    with open(path_pred_second_action_dict, "r") as f:
        pred_second_action_dicts = json.load(f)

    df_result = pd.DataFrame(columns=["video_path", "total_acc", "total_acc_wo_none"])
    video_paths, total_accs, total_accs_wo_none = [], [], []
    
    for video_path in tqdm(pred_second_action_dicts.keys()):
        if video_path not in gt_second_action_dicts:
            print(f"{video_path} is not in the ground-truth file.")
            continue
        gt_second_action_dict = gt_second_action_dicts[video_path]
        pred_second_action_dict = pred_second_action_dicts[video_path]

        if abs(len(gt_second_action_dict) - len(pred_second_action_dict)) > 1:
            print(f"The number of seconds in the ground-truth and prediction files are quite different. | video_path: {video_path}")

        sorted_gt_seconds = sorted(gt_second_action_dict.keys(), key=int)
        sorted_pred_seconds = sorted(pred_second_action_dict.keys(), key=int)
        
        gt_actions = []
        pred_actions = []
        total_secs = len(sorted_gt_seconds)
        target_secs = 0
        correct_secs_none = 0
        correct_secs = 0

        for i in range(len(sorted_gt_seconds)):
            gt_second = sorted_gt_seconds[i]
            if gt_second not in pred_second_action_dict:
                continue
            pred_second = sorted_pred_seconds[i]

            pred_action = pred_second_action_dict[pred_second]
            if pred_action != "none":
                if pred_action not in dataset_actions:
                    print(f"Unknown action: {pred_action}")
                    pred_action = "none"
            pred_actions.append(pred_action)
            
            gt_action = gt_second_action_dict[gt_second]
            gt_actions.append(gt_action)
            
            if gt_action == "none":
                if pred_action == gt_action:
                    correct_secs_none += 1
            else:
                if pred_action == gt_action:
                    correct_secs += 1
                target_secs += 1
        
        total_acc = (correct_secs+correct_secs_none) / total_secs
        if target_secs == 0:
            total_acc_wo_none = 0
        else:
            total_acc_wo_none = correct_secs / target_secs
        video_paths.append(video_path)
        total_accs.append(total_acc)
        total_accs_wo_none.append(total_acc_wo_none)

        if save_fig:
            save_path = os.path.join(output_folder_path, os.path.basename(path_pred_second_action_dict).replace(".json", "_")+os.path.basename(video_path).replace(".mp4", ".png"))
            plot_action_bars(gt_actions, pred_actions, save_path=save_path, color_map=color_map, legend_classes=dataset_actions)
    
    df_result["video_path"] = video_paths
    df_result["total_acc"] = total_accs
    df_result["total_acc_wo_none"] = total_accs_wo_none
    print(f"Average Total Accuracy: {df_result['total_acc'].mean()}")
    print(f"Average Total Accuracy (w/o 'none'): {df_result['total_acc_wo_none'].mean()}")

    if save:
        model_name = path_pred_second_action_dict.split("/")[-1].replace("predictions_", "").replace(".json", "")
        save_path = os.path.join(output_folder_path, f"coin_results_{model_name}.csv")
        df_result.to_csv(save_path, index=False)

def visualize_results(path_gt_second_action_dict, path_pred_second_action_dict, output_folder_path="./output", save=True):
    with open(path_gt_second_action_dict, "r") as f:
        annotations = json.load(f)
    with open(path_pred_second_action_dict, "r") as f:
        predictions = json.load(f)
    model_name = os.path.basename(path_pred_second_action_dict).replace("predictions_", "").replace(".json", "")
    os.makedirs(os.path.join(output_folder_path, model_name), exist_ok=True)

    GREEN = (0, 255, 0)
    RED = (0, 0, 255)
    WHITE = (255, 255, 255)

    # dataset_action_conversion = {"install legs of sofa": "install",
    #                           "put on sofa cover": "put",
    #                           "place cushion and backrest": "place",
    #                           "lift the barbell directly over the head": "lift",
    #                           "stand up with straight legs": "none",
    #                           "pull the barbell to the chin": "pull",
    #                           "lift the barbell from the chin over the head": "lift",
    #                           "place license plate cover": "place",
    #                           "screw and fix the cover": "none",
    #                           "place license plate": "place",
    #                           "assemble the frame": "none",
    #                           "install vertical boards": "install",
    #                           "install horizontal boards": "install",
    #                           "place the board on each side": "place",
    #                           "put all bed boxes together": "put",
    #                           "place the bed board": "place",
    #                           "tile the wall": "none",
    #                           "wipe adhesive and bedding layer": "none",
    #                           "load the inner tube": "load",
    #                           "load the tire": "load",
    #                           "remove the light socket and disconnect the wire": "remove",
    #                           "install the light socket": "install",
    #                           "install the bulb and light housing or shell": "install",
    #                           "unscrew the screw": "none",
    #                           "jack up the car": "none",
    #                           "squat and hold the barbell": "hold",
    #                           "put down the barbell": "put",
    #                           "cut the old chain": "cut",
    #                           "take old chain away": "take",
    #                           "install head of the bed": "install",
    #                           "place board at the bottom": "place",
    #                           "place mattress on the bed": "place",
    #                           "install the wheels for the base": "install",
    #                           "assemble the cushion and the backrest": "none",
    #                           "connect the chair and the base": "none",
    #                           "take out the old filter": "take",
    #                           "remove the cap of the new filter": "remove",
    #                           "install the new filter": "install",
    #                           "install armrest on sofa": "install",
    #                           "install the new bulb": "install",
    #                           "install the light shell/housing/support": "install",
    #                           "remove the tire": "remove",
    #                           "pump up the tire": "none",
    #                           "fit on the boards": "none",
    #                           "cut the raw boards": "cut",
    #                           "put on the tire": "put",
    #                           "tighten screws": "tighten",
    #                           "remove the inner tube": "remove",
    #                           "put every parts mentioned together": "put",
    #                           "install the sofa back": "install",
    #                           "install the new chain": "install",
    #                           "take out the shell": "take",
    #                           "take out the filter": "take",
    #                           "fit on the filter": "none",
    #                           "fit on the shell": "none",
    #                           "remove inner tube": "remove",
    #                           "put inner tube back": "put",
    #                           "clean the scratch": "none",
    #                           "apply polishing paste on the surface": "apply",
    #                           "remove the light housing or shell and bulb": "remove",
    #                           "install legs on the bed": "install",
    #                           "mount the bracket to the back of the car": "none",
    #                           "place the bicycle on the bracket": "place",
    #                           "fasten the strap": "none",
    #                           "knock in the nails": "none",
    #                           "prepare adhesive": "none",
    #                           "check ground and layout planning": "none",
    #                           "fill the gap": "none",
    #                           "prepare the ground floor": "none",
    #                           "unload the wheel": "unload",
    #                           "load the wheel": "load",
    #                           "open the fuel tank cap": "open",
    #                           "insert oil gun in the car": "insert",
    #                           "operate the control panel": "none",
    #                           "pullthe  oil gun out": "pull",
    #                           "close the fuel tank cap": "close",
    #                           "connect the wire again": "none",
    #                           "look for leaks": "none",
    #                           "use sandpaper/metal to polish rubber near leak": "none",
    #                           "apply glue": "apply",
    #                           "paste patch": "none",
    #                           "install window panel": "install",
    #                           "connect air conditioners and windows": "none",
    #                           "polish with polisher": "none",
    #                           "take out the battery": "take",
    #                           "unscrew the screws used to fix the screen": "none",
    #                           "pull out the screen connector and remove the screen": "pull",
    #                           "wrap the pipe band": "none",
    #                           "install the new shower head": "install",
    #                           "apply some glue on the borads": "apply",
    #                           "fix and fasten the upper and lower ropes to car": "none",
    #                           "open the laptop rear cover": "open",
    #                           "remove the old hard disk": "remove",
    #                           "install stand of the seat": "install",
    #                           "install the air nozzle": "install",
    #                           "pump up to the tire": "none",
    #                           "remove the air nozzle": "remove",
    #                           "place crossbar on shelves": "place",
    #                           "install two shelves": "install",
    #                           "hang up curtains": "none",
    #                           "open cover": "open",
    #                           "remove battery": "remove",
    #                           "open the car key cover": "open",
    #                           "take out the car key battery": "take",
    #                           "put in the battery": "put",
    #                           "close the car key cover": "close",
    #                           "draw lines to mark the hole": "none",
    #                           "drill with an electric drill": "none",
    #                           "install fan tray": "install",
    #                           "install fans and lights": "install",
    #                           "remove surface of the door": "remove",
    #                           "remove old glass from window lift": "remove",
    #                           "cut branches": "cut",
    #                           "connect branches": "none",
    #                           "glue on window frame": "none",
    #                           "install new glass on window lift": "install",
    #                           "tighten the valve and screw on the valve cap": "tighten",
    #                           "fix window board": "none",
    #                           "pull out a lock lever at bottom of gun": "pull",
    #                           "put in a new grease container": "put",
    #                           "open the lid of new container": "open",
    #                           "screw gun head back": "none",
    #                           "try to press gun head, spray residual old grease": "none",
    #                           "replace the old shower head": "none",
    #                           "install the door": "install",
    #                           "screw off the valve cap and open the valve": "none",
    #                           "hold electrode with a welding tong": "hold",
    #                           "wear a eye protection helmet": "none",
    #                           "weld along the weld line": "none",
    #                           "clean weld line with a hammer": "none",
    #                           "open up the cover": "open",
    #                           "take out the fuse": "take",
    #                           "fit on the fuse": "none",
    #                           "clean the surface": "none",
    #                           "wind the junction to fasten the connection": "none",
    #                           "install fan bracket": "install",
    #                           "pick up the back cover of the phone with the cymbal": "none",
    #                           "remove the components of the fixed battery": "remove",
    #                           "take down the old battery": "take",
    #                           "close up the cover": "close",
    #                           "cut tiles": "cut",
    #                           "remove the old wax ring and clean": "remove",
    #                           "install the new wax ring": "install",
    #                           "find the position of the hole": "none",
    #                           "apply toothpaste or other cleaning agent to scratch": "apply",
    #                           "wipe the scratch with cloth": "none",
    #                           "wipe the toothpaste": "none",
    #                           "install the closestool again": "install",
    #                           "remove the light shell/housing/support": "remove",
    #                           "take out the old bulb": "take",
    #                           "install fan frame": "install",
    #                           "connect the water pipe": "none",
    #                           "none": "none"}
    # dataset_actions = list(set(list(dataset_action_conversion.values())))
    # dataset_actions = pd.read_csv("./COIN_videos/filtered_COIN_videos/task_step.csv")["Step"].unique().tolist()
    dataset_actions = pd.read_csv("./COIN_videos/filtered_COIN_videos/task_step.csv")["unified_2"].unique().tolist()
    if "none" not in dataset_actions:
        dataset_actions.append("none")
    
    df_dataset_actions = pd.read_csv("./COIN_videos/filtered_COIN_videos/task_step.csv")
    dataset_action_conversion = {}
    for i in range(len(df_dataset_actions)):
        key = "---".join([df_dataset_actions.iloc[i]["Task"], df_dataset_actions.iloc[i]["Step"]])
        value = df_dataset_actions.iloc[i]["unified_2"]
        dataset_action_conversion[key] = value

    for video_path in tqdm(predictions.keys()):
        if video_path not in annotations:
            print(f"{video_path} is not in the ground-truth file.")
            continue
        gt_labels = annotations[video_path]
        pred_labels = predictions[video_path]

        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        output_path = os.path.join(output_folder_path, model_name, os.path.basename(video_path))
        out = cv2.VideoWriter(output_path, fourcc, int(fps), (width, height))

        frame_idx = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            seconds = int(frame_idx // fps)
            gt_label = gt_labels.get(str(seconds), "none")
            
            # if gt_label != "none":
            #     video_name = os.path.basename(video_path).replace(".mp4", "")
            #     key = video_name.split("_")[0] + "---" + gt_label
            #     gt_label = dataset_action_conversion[key]

            pred_label = pred_labels.get(str(seconds), "none")
            if pred_label != "none":
                # pred_label = pred_label.split(" - ")[-1]
                if pred_label not in dataset_actions:
                    print(f"Unknown action: {pred_label}")

            # color = GREEN if (gt_label == pred_label) or (dataset_action_conversion[gt_label] == dataset_action_conversion[pred_label]) else RED
            color = GREEN if gt_label == pred_label else RED

            y_offset = 20
            # get black box for text
            text_size, _ = cv2.getTextSize(f"GT: {gt_label}", cv2.FONT_HERSHEY_SIMPLEX, 0.8, 1)
            text_w, text_h = text_size
            # cv2.rectangle(frame, (0, 0), (width, y_offset*3), (0, 0, 0), -1)
            cv2.rectangle(frame, (0, 0), (int(1.1*text_w), y_offset+int(1.1*text_h)), (0, 0, 0), -1)
            cv2.putText(frame, f"GT: {gt_label}", (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.8, WHITE, 2)
            cv2.putText(frame, f"Pred: {pred_label}", (10, y_offset*2), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
            out.write(frame)
            frame_idx += 1

        cap.release()
        out.release()
    return

if __name__ == "__main__":
    # path_gt_second_action_dict = "./ava/train_preprocessed/annotation.json"
    # path_pred_second_action_dict = "./ava/train_preprocessed/predictions_gpt-4o.json"
    # testing_ava(path_gt_second_action_dict, path_pred_second_action_dict, output_folder_path="./output", save=True, save_fig=True)
    
    # path_gt_second_action_dict = "./COIN_videos/filtered_COIN_videos/annotation.json"
    path_gt_second_action_dict = "./COIN_videos/filtered_COIN_videos/annotation_gunwoo.json"

    # path_pred_second_action_dict = "./COIN_videos/filtered_COIN_videos/predictions_gpt-4o_1.json"
    # path_pred_second_action_dict = "./COIN_videos/filtered_COIN_videos/predictions_gpt-4o_2.json"
    path_pred_second_action_dict = "./COIN_videos/filtered_COIN_videos/predictions_gpt-4o_4.json"
    # path_pred_second_action_dict = "./COIN_videos/filtered_COIN_videos/predictions_gemini-1.5-pro_1.json"
    # path_pred_second_action_dict = "./COIN_videos/filtered_COIN_videos/predictions_gemini-2.0-flash-exp_1.json"
    # path_pred_second_action_dict = "./COIN_videos/filtered_COIN_videos/predictions_gemini-2.0-flash_1.json"
    # path_pred_second_action_dict = "./COIN_videos/filtered_COIN_videos/predictions_claude-3-5-sonnet-latest_1.json"
    # testing_coin(path_gt_second_action_dict, path_pred_second_action_dict, output_folder_path="./output", save=True, save_fig=True)
    # testing_coin_v2(path_gt_second_action_dict, path_pred_second_action_dict, output_folder_path="./output", save=True, save_fig=True)
    # testing_coin_v3(path_gt_second_action_dict, path_pred_second_action_dict, output_folder_path="./output", save=True, save_fig=True)
    testing_coin_v4(path_gt_second_action_dict, path_pred_second_action_dict, output_folder_path="./output", save=True, save_fig=True)
    visualize_results(path_gt_second_action_dict, path_pred_second_action_dict, output_folder_path="./output", save=True)

    
    # txt = """
    #     [
    #     {
    #         "time": "0:54",
    #         "text": "put down the barbell"
    #     },
    #     {
    #         "time": "0:55",
    #         "text": "lift the barbell directly over the head"
    #     },
    #     {
    #         "time": "0:57",
    #         "text": "put down the barbell"
    #     },
    #     {
    #         "time": 

    #     """
    # import utils
    # fixed_txt = utils.fix_broken_json(txt)
    # print(fixed_txt)
"""
Invalid action prediction: push the castors onto the wheel base
Invalid action prediction: push the gas lift into the large aperture beneath the chair
Invalid action prediction: lift the chair and place the gas lift into the base
Invalid action prediction: make sure that the seat and gas lift are securely positioned
Invalid action prediction: wind up handle to make the tilting action harder or easier to move
Invalid action prediction: pull out for easier operation and pushed in when not being used
Invalid action prediction: pull up to release the chair height adjustment
Invalid action prediction: flip up to release the tilt action
Invalid action prediction: remove the batteries
Invalid action prediction: remove the valve cap
Invalid action prediction: place the tire on the wheel
Invalid action prediction: pump up the tube
Invalid action prediction: open up the filter
Invalid action prediction: take out your old filter
Invalid action prediction: slide it in
Invalid action prediction: grind the floor
Invalid action prediction: pour water on the floor
Invalid action prediction: use a floor grinder
Invalid action prediction: pour floor leveling compound
Invalid action prediction: apply the material
Invalid action prediction: lay the plastic
Invalid action prediction: mark a line
Invalid action prediction: lay the floor
Invalid action prediction: cut the boards
Invalid action prediction: check the boards
Invalid action prediction: use my coin
Invalid action prediction: rotate the screw that holds the battery in place
Invalid action prediction: lift out
Invalid action prediction: use my Phillips double zero screwdriver
Invalid action prediction: loosen those up
Invalid action prediction: gently pull the memory door
Invalid action prediction: pull tab
Invalid action prediction: pull the pull tab
Invalid action prediction: slides right out
Invalid action prediction: remove these screws
Invalid action prediction: take off the bracket
Invalid action prediction: reattach this bracket to our new hard drive
Invalid action prediction: placing the bike rack up onto the vehicle’s rear hatch
Invalid action prediction: take the top clips to put to the top part of the rear hatch
Invalid action prediction: side clips to the side part of the rear hatch
Invalid action prediction: bottom clips to the bottom part of the rear hatch
Invalid action prediction: tighten down the straps to secure the bike rack to the cargo door
Invalid action prediction: place our bike into the frame cradles
"""