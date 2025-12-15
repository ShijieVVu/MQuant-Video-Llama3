from ..smp import *
from .video_base import VideoBaseDataset
import re
import json

try:
    import decord
except ImportError:
    warnings.warn("Please install decord via `pip install decord`.")

FAIL_MSG = "Failed to obtain answer via API."


class VideoEOC(VideoBaseDataset):

    TYPE = "MCQ"

    def __init__(self, dataset="VideoEOC", use_subtitle=False):
        super().__init__(dataset=dataset)
        self.use_subtitle = use_subtitle

    @classmethod
    def supported_datasets(cls):
        return ["VideoEOC"]

    def prepare_dataset(self, dataset_name="VideoEOC", repo_id=None):
        """
        Prepare the VideoEOC dataset from local files.
        Expected structure:
        - dataset_root/
          - videos/
            - video1.mp4
            - video2.mp4
          - meta_infos.json or meta_infos_subset.json
        """
        # For VideoEOC, we expect the dataset to be provided locally
        # Users should set LMU_ROOT environment variable or provide the path
        lmu_root = LMUDataRoot()
        dataset_root = osp.join(lmu_root, dataset_name)
        
        # Check for annotation file
        ann_file = None
        for possible_name in ["meta_infos.json", "meta_infos_subset.json"]:
            possible_path = osp.join(dataset_root, possible_name)
            if osp.exists(possible_path):
                ann_file = possible_path
                break
        
        if ann_file is None:
            raise FileNotFoundError(
                f"Could not find annotation file in {dataset_root}. "
                f"Please provide meta_infos.json or meta_infos_subset.json"
            )
        
        # Load and convert JSON to TSV format expected by base class
        with open(ann_file, 'r') as f:
            data = json.load(f)
        
        # Convert to DataFrame format
        rows = []
        for item in data:
            row = {
                'index': item['idx'],
                'video': item['video_path'].replace('.mp4', ''),  # Remove extension
                'video_path': osp.join('videos', item['video_path']),
                'question': item['question'],
                'answer': json.dumps(item['answer']),  # Store as JSON string
                'video_type': item['video_type'],
                'video_time': item.get('video_time', 0),
                'fps': item.get('fps', 30),
                'frame_number': item.get('frame_number', 0),
                'box': json.dumps(item.get('box', [])),  # Store as JSON string
                'choice_type': item.get('choice_type', 'single-choice'),
            }
            # Add choices if present
            if 'choices' in item:
                row['choices'] = json.dumps(item['choices'])
            rows.append(row)
        
        df = pd.DataFrame(rows)
        
        # Save as TSV
        tsv_file = osp.join(dataset_root, f"{dataset_name}.tsv")
        df.to_csv(tsv_file, sep='\t', index=False)
        
        return dict(data_file=tsv_file, root=dataset_root)

    def save_video_frames(self, video, target_fps=1):
        """Extract frames from video file at specified FPS.
        Mimics EOCBench's get_index behavior for fps_segments mode.
        
        Args:
            video: Video identifier (without extension)
            target_fps: Target frame rate for sampling (frames per second). Default is 1 fps.
        
        Returns:
            frame_paths: List of paths to extracted frames
            indices: List of frame indices extracted from the video
            video_info: Dictionary with video metadata
        """
        # Ensure video is a string (pandas may read it as int from TSV)
        video = str(video)
        vid_path = osp.join(self.data_root, "videos", video + ".mp4")
        
        if not osp.exists(vid_path):
            raise FileNotFoundError(f"Video file not found: {vid_path}")
        
        vid = decord.VideoReader(vid_path)
        video_fps = vid.get_avg_fps()
        max_frame = len(vid) - 1
        
        # Calculate sampling interval (frames between samples)
        # This matches EOCBench: n_frames = fps // fps_segments
        n_frames = int(video_fps / target_fps)
        
        # Cap at 150 frames interval (EOCBench behavior)
        n_frames = min(n_frames, 150)
        
        # Handle edge case: if interval is 0, sample all frames
        if n_frames == 0:
            indices = list(range(max_frame + 1))
        else:
            # Sample frames at regular intervals, mimicking EOCBench's get_index
            indices = []
            for frame_count in range(max_frame + 1):
                # Sample if divisible by interval OR it's the last frame
                if frame_count % n_frames == 0 or frame_count == max_frame:
                    indices.append(frame_count)
        
        num_frames = len(indices)

        video_info = {
            "fps": video_fps,
            "n_frames": max_frame + 1,
            "target_fps": target_fps,
            "sampling_interval": n_frames,
            "extracted_frames": num_frames,
        }

        frame_paths = self.frame_paths(video, num_frames)
        flag = np.all([osp.exists(p) for p in frame_paths])

        if not flag:
            images = [vid[i].asnumpy() for i in indices]
            images = [Image.fromarray(arr) for arr in images]
            for im, pth in zip(images, frame_paths):
                if not osp.exists(pth):
                    im.save(pth)

        return frame_paths, indices, video_info

    def build_prompt(self, line, target_fps=1):
        """Build prompt for VideoEOC dataset.
        
        Args:
            line: Data row containing video and question information
            target_fps: Target frame rate for sampling (frames per second). Default is 1 fps.
        """
        if isinstance(line, int):
            assert line < len(self)
            line = self.data.iloc[line]

        frames, indices, video_info = self.save_video_frames(line["video"], target_fps)
        
        # Parse stored JSON fields
        answer = json.loads(line['answer']) if isinstance(line['answer'], str) else line['answer']
        boxes = json.loads(line['box']) if isinstance(line['box'], str) else line['box']
        
        # Handle choices carefully - check for NaN values
        raw_choices = None
        if 'choices' in line and pd.notna(line['choices']):
            if isinstance(line['choices'], str):
                try:
                    raw_choices = json.loads(line['choices'])
                except:
                    raw_choices = None
            else:
                raw_choices = line['choices']
        
        # Build the prompt text
        question = line['question']
        colors = ['red', 'blue', 'green', 'yellow', 'purple', 'orange', 'pink']
        
        # Check if question has object tags
        pattern = r'<object \d>'
        matches = re.findall(pattern, question)
        
        # Construct prompt based on video type and object presence (matching EOCBench)
        if line['video_type'] == 'Absolute Time Perception':
            prompt_str = f"I have overlaid the box on the last frame of the video, "
            for i in range(len(matches)):
                prompt_str += f"<object {i}>:{colors[i]}; "
            prompt_str = prompt_str[:-2] + '. '
            
            prompt = f"""
Question: {prompt_str}{question} Please output the answer directly in seconds.
"""
        
        elif len(matches) > 0:
            prompt_str = f"I have overlaid the box on the last frame of the video, "
            for i in range(len(matches)):
                prompt_str += f"<object {i}>:{colors[i]}; "
            prompt_str = prompt_str[:-2] + '. '
            
            if raw_choices:
                choices = [f"{option.upper()}. {choice}" for option, choice in raw_choices.items()]
                options = "\n".join(choices)
            else:
                options = ""
            
            prompt = f"""
Question: {prompt_str}{question}
Options: 
{options}
"""
            if len(answer) == 1:
                prompt += "Answer directly using the letters of the options given and wrap your response."
            else:
                prompt += "Answer directly using the letters of the options given. There are multiple answers, so wrap your response in <choice></choice>. For example, if the answer is A and B, then output <choice>A, B</choice>; if the answer is A, B and C, then output <choice>A, B, C</choice>"
        
        else:
            if raw_choices:
                choices = [f"{option.upper()}. {choice}" for option, choice in raw_choices.items()]
                options = "\n".join(choices)
            else:
                options = ""
                
            prompt = f"""
Question: {question}
Options: 
{options}
"""
            if len(answer) == 1:
                prompt += "Answer directly using the letters of the options given and wrap your response in <choice></choice>. For example, if the answer is A, then output <choice>A</choice>"
            else:
                prompt += "Answer directly using the letters of the options given. There are multiple answers, so wrap your response in <choice></choice>. For example, if the answer is A and B, then output <choice>A, B</choice>; if the answer is A, B and C, then output <choice>A, B, C</choice>"
        
        # Build message format
        message = []

        # Add each frame as an image
        for frame_path in frames:
            message.append(dict(type="image", value=frame_path))

        message.append(dict(type="text", value=prompt))
        
        return message

    @classmethod
    def evaluate(cls, eval_file, **judge_kwargs):
        """Evaluate predictions for VideoEOC dataset."""
        assert eval_file.endswith(".xlsx"), "data file should be an xlsx file"
        
        from .utils.videoeoc_eval import calculate_metrics
        
        data = load(eval_file)
        
        # Calculate metrics
        metrics = calculate_metrics(data)
        
        # Save results
        score_file = eval_file.replace(".xlsx", "_score.xlsx")
        rating_file = eval_file.replace(".xlsx", "_rating.json")
        
        dump(data, score_file)
        dump(metrics, rating_file)
        
        return metrics
