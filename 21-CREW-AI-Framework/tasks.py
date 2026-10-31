
from crewai import Task

from agents import blog_researcher, blog_writer


research_task = Task(
    description=(
        "Research the topic: {topic}.\n"
        "Use the available YouTube search tool to find relevant "
        "videos from the configured channel.\n"
        "Summarize the relevant concepts, explanations, examples, "
        "and important distinctions found in the retrieved content.\n"
        "Include video titles or source details when available.\n"
        "Do not invent video content or claim that a video covers "
        "something unless the retrieved information supports it."
    ),
    expected_output=(
        "A structured research report containing:\n"
        "1. Relevant video and source details.\n"
        "2. A clear explanation of the topic.\n"
        "3. Important concepts and comparisons.\n"
        "4. Examples and practical observations.\n"
        "5. Any gaps or uncertainties in the retrieved information."
    ),
    agent=blog_researcher,
)


write_task = Task(
    description=(
        "Write a technical blog about {topic} using the completed "
        "research report from the preceding task.\n"
        "Explain the concepts in beginner-friendly language.\n"
        "Use a suitable title, introduction, descriptive headings, "
        "examples, comparisons, and a conclusion.\n"
        "Keep the article technically accurate and avoid unsupported "
        "claims. Do not invent quotations or source citations."
    ),
    expected_output=(
        "A polished Markdown blog post with a title, introduction, "
        "well-organized sections, examples, a conclusion, and "
        "source details when available."
    ),
    agent=blog_writer,
    context=[research_task], ##It tells CrewAI that the writing task depends on the research task's output. The writer can use that output as context instead of starting from scratch.
    output_file="new-blog-post.md",
)
