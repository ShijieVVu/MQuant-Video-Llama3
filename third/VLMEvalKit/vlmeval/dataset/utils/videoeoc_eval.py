"""
Evaluation utilities for VideoEOC dataset.
Adapted from EOCBench evaluation logic.
"""
import re
import json
import difflib
from collections import defaultdict


def get_content_between_a_b(start_tag, end_tag, text):
    """Extract text between tags."""
    extracted_text = ""
    start_index = text.find(start_tag)
    while start_index != -1:
        end_index = text.find(end_tag, start_index + len(start_tag))
        if end_index != -1:
            extracted_text += text[start_index + len(start_tag) : end_index] + " "
            start_index = text.find(start_tag, end_index + len(end_tag))
        else:
            break
    return extracted_text.strip()


def extract(text, type_tag, hard=True):
    """Extract content from XML-like tags."""
    if text:
        target_str = get_content_between_a_b(f"<{type_tag}>", f"</{type_tag}>", text)
        if target_str:
            return target_str
        elif hard:
            return text
        else:
            return ""
    else:
        return ""


def find_first_number(s):
    """Find the first number in a string."""
    pattern = r'\d+\.?\d*'
    match = re.search(pattern, s)
    if match:
        return match.group()
    else:
        return None


def extract_time(text):
    """Extract time from text, removing object tags first."""
    def replace_object_tags(text):
        result = re.sub(r'<object \d+>', '', text)
        return result
    
    text = replace_object_tags(text)
    number = find_first_number(text)
    if number is None:
        return 0
    return number


def calculate_time_awareness_score(gt, pred, thresholds=None):
    """Calculate time awareness score with different error thresholds."""
    gt, pred = float(gt), float(pred)
    errors = [0.01, 0.1, 0.2, 0.3]
    accurate_counts = [0] * len(errors)
    
    error = abs(gt - pred)
    
    for i, threshold in enumerate(errors):
        if error <= threshold * gt:
            accurate_counts[i] += 1

    score = sum(accurate_counts) / len(errors)
    return score


def str_similarity(str1, str2):
    """Calculate string similarity."""
    seq = difflib.SequenceMatcher(None, str1, str2)
    return seq.ratio()


def find_most_similar_index(str_list, target_str):
    """Find the index of the most similar string in the list."""
    most_similar_index = 0
    highest_similarity = 0

    for i, s in enumerate(str_list):
        similarity = str_similarity(s, target_str)
        
        if similarity > highest_similarity:
            most_similar_index = i
            highest_similarity = similarity
    
    return most_similar_index


def calculate_metrics(data):
    """
    Calculate evaluation metrics for VideoEOC dataset.
    
    Args:
        data: DataFrame or dict with columns/keys: 
            - prediction: model output
            - answer: ground truth answer(s)
            - video_type: type of question
            - choices: answer choices (optional)
            - choice_type: single-choice, multi-choice, or open-ended
    
    Returns:
        Dictionary of metrics organized by video_type and temporal category
    """
    # Mapping from video types to temporal categories
    key_mapping = {
        "Object State Retrospection": "Past",
        "Location Retrospection": "Past",
        "Object Relationship Evolution": "Past",
        "Absolute Time Perception": "Past",
        "Immediate State Recognition": "Present",
        "Object Relationship": "Present",
        "Purpose and Function Inference": "Present",
        "Anomaly Perception": "Present",
        "Trajectory and Motion Prediction": "Future",
        "State Change Prediction": "Future",
        "Dynamic Relationship Prediction": "Future"
    }
    
    total_cnt = defaultdict(float)
    total_right = defaultdict(float)
    total_single = defaultdict(float)
    total_multi = defaultdict(float)
    single_right = defaultdict(float)
    multi_right = defaultdict(float)
    adjust_right = defaultdict(float)
    total_failed = defaultdict(float)
    
    # Convert data to list of dicts if it's a DataFrame
    if hasattr(data, 'to_dict'):
        samples = data.to_dict('records')
    else:
        samples = data
    
    for i, sample in enumerate(samples):
        name = sample["video_type"]
        
        # Parse answer
        answer = sample["answer"]
        if isinstance(answer, str):
            try:
                answer = json.loads(answer)
            except:
                answer = [answer]
        answer = [item.lower() for item in answer]
        
        # Get prediction
        response = sample.get("prediction", "")
        success = sample.get("success", True)
        
        if isinstance(response, list):
            response = "a"
        response = response.lower()
        orig_response = response
        
        if "<s>" in response:
            response = extract(response, "s")
        
        alphas = ["a", "b", "c", "d", "e"]
        
        # Parse choices if present
        choices = sample.get("choices", {})
        if isinstance(choices, str):
            try:
                choices = json.loads(choices)
            except:
                choices = {}
        
        choice_type = sample.get('choice_type', 'single-choice')
        
        try:
            if choice_type == "open-ended":
                response = extract_time(response)
            else:
                response = extract(response, "choice")
                if "." in response:
                    response = response.split(".")[0]
                response = [item.strip() for item in response.split(",")]
                
                # Validate response
                for r in response:
                    if r not in alphas:
                        if choices:
                            options = [f"{key}. {value}" for key, value in choices.items()]
                            response = find_most_similar_index(options, orig_response)
                            response = [alphas[response]]
                        else:
                            response = ["a"]
                        break
        except:
            response = ["a"]

        if not success:
            total_failed[name] += 1
            total_failed[key_mapping.get(name, "Unknown")] += 1
            total_failed["total"] += 1

        # Count single vs multi-choice
        if len(answer) == 1:
            total_single["total"] += 1
            total_single[name] += 1
            total_single[key_mapping.get(name, "Unknown")] += 1
        else:
            total_multi["total"] += 1
            total_multi[name] += 1
            total_multi[key_mapping.get(name, "Unknown")] += 1
        
        # Calculate scores
        if choice_type == 'open-ended':
            time_answer = find_first_number(answer[0])
            if time_answer:
                time_score = calculate_time_awareness_score(time_answer, response)
                total_right["total"] += time_score
                total_right[name] += time_score
                total_right[key_mapping.get(name, "Unknown")] += time_score
        else:
            if sorted(response) == sorted(answer):
                total_right["total"] += 1
                total_right[name] += 1
                total_right[key_mapping.get(name, "Unknown")] += 1
                
                if len(answer) == 1:
                    single_right["total"] += 1
                    single_right[name] += 1
                    single_right[key_mapping.get(name, "Unknown")] += 1
                else:
                    multi_right["total"] += 1
                    multi_right[name] += 1
                    multi_right[key_mapping.get(name, "Unknown")] += 1
        
            # Calculate partial credit
            response = [option.lower() for option in response]
            answer = [option.lower() for option in answer]
            partial_right = 0
            for option in response:
                if option in answer:
                    partial_right += 1/(len(answer))
                else:
                    partial_right = 0
                    break
            adjust_right[name] += partial_right
            adjust_right[key_mapping.get(name, "Unknown")] += partial_right
            adjust_right["total"] += partial_right
        
        total_cnt["total"] += 1
        total_cnt[name] += 1
        total_cnt[key_mapping.get(name, "Unknown")] += 1
    
    # Calculate final scores
    final_scores = {}
    for name in total_cnt.keys():
        final_scores[name] = {
            "total_acc": total_right[name]/total_cnt[name],
            "single_acc": single_right[name]/(total_single[name]+1e-6),
            "multi_acc": multi_right[name]/(total_multi[name]+1e-6),
            "adjust_acc": adjust_right[name]/total_cnt[name],
            "total_cnt": total_cnt[name],
            "right_cnt": total_right[name],
            "single_cnt": total_single[name],
            "right_single_cnt": single_right[name],
            "multi_right_cnt": multi_right[name],
            "multi_cnt": total_multi[name],
            "failed_cnt": total_failed[name]
        }
    
    return final_scores

