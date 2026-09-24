from langgraph.graph import StateGraph, START, END

from .state import YouTubeLearningState

from .nodes import (
    load_transcript,
    analyze_video,
    consolidate_topics,
    select_topic,
    teach_topic,
    generate_quiz,
    evaluate_quiz,
    teach_again,
    next_topic
)


# ============================================================
# CREATE GRAPH
# ============================================================

graph = StateGraph(YouTubeLearningState)

# ============================================================
# ADD NODES
# ============================================================

graph.add_node(
    "load_transcript",
    load_transcript
)

graph.add_node(
    "analyze_video",
    analyze_video
)

graph.add_node(
    "consolidate_topics",
    consolidate_topics
)

graph.add_node(
    "select_topic",
    select_topic
)

graph.add_node(
    "teach_topic",
    teach_topic
)

graph.add_node(
    "generate_quiz",
    generate_quiz
)

graph.add_node(
    "evaluate_quiz",
    evaluate_quiz
)

graph.add_node(
    "teach_again",
    teach_again
)

graph.add_node(
    "next_topic",
    next_topic
)

# ============================================================
# NORMAL EDGES
# ============================================================

graph.add_edge(
    START,
    "load_transcript"
)

graph.add_edge(
    "load_transcript",
    "analyze_video"
)

graph.add_edge(
    "analyze_video",
    "consolidate_topics"
)

graph.add_edge(
    "consolidate_topics",
    "select_topic"
)

graph.add_edge(
    "teach_topic",
    "generate_quiz"
)

graph.add_edge(
    "generate_quiz",
    "evaluate_quiz"
)


# ============================================================
# CONDITIONAL ROUTING
# ============================================================

def route_after_evaluation(state):

    return state["next_action"]


graph.add_conditional_edges(
    "evaluate_quiz",
    route_after_evaluation,
    {
        "teach_again": "teach_again",
        "next_topic": "next_topic"
    }
)


# ============================================================
# LOOP
# ============================================================

graph.add_edge(
    "teach_again",
    "teach_topic"
)

graph.add_edge(
    "next_topic",
    "select_topic"
)


# ============================================================
# FINISH
# ============================================================

graph.add_conditional_edges(
    "select_topic",
    lambda state: state["next_action"],
    {
        "teach": "teach_topic",
        "finish": END
    }
)


# ============================================================
# COMPILE
# ============================================================

app = graph.compile()


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":

    result = app.invoke({

        "youtube_url":
        "https://www.youtube.com/watch?v=_IPP7_Bi8uA",

        "level":
        "College Student",

        "goal":
        "Understand concepts",

        "transcript": "",

        "topics": [],

        "current_topic_index": 0,

        "current_topic": {},

        "lesson": "",

        "quiz": [],

        "score": 0,

        "next_action": ""

    })

    print("\n================================")
    print("YOUTUBE LEARNING AGENT COMPLETE")
    print("================================")

    print("\nFinal Score:")
    print(result["score"])

    print("\nTopics:")
    print(result["topics"])