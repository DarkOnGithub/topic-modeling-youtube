import os
import json
from typing import List

def load_comments_from_folder(channel_folder: str, data_root: str = 'data') -> List[str]:
    """
    Load all comments from JSON files for a specific channel.
    """
    comments = []
    videos_dir = os.path.join(data_root, channel_folder, 'videos')
    
    if not os.path.exists(videos_dir):
        return []

    for filename in os.listdir(videos_dir):
        if filename.endswith('.json'):
            filepath = os.path.join(videos_dir, filename)
            with open(filepath, 'r', encoding='utf-8') as f:
                try:
                    video_data = json.load(f)
                    for comment in video_data.get('comments', []):
                        if comment.get('text'):
                            comments.append(comment['text'])
                except Exception as e:
                    print(f"Error while reading {filepath}: {e}")
    
    return comments

