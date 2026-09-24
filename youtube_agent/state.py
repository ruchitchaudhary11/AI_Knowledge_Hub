from typing import TypedDict, List, Dict


class YouTubeLearningState(TypedDict):

    youtube_url: str

    level: str

    goal: str

    transcript: str

    topics: List[Dict]

    current_topic_index: int

    current_topic: Dict

    lesson: str

    quiz: List[Dict]

    topic_context: str

    score: int

    next_action: str