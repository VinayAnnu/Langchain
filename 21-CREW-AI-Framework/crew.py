## The crew is the orchestration layer. It brings agents and tasks together and starts execution.
import truststore
truststore.inject_into_ssl()

from crewai import Crew, Process

from agents import blog_researcher, blog_writer
from tasks import research_task, write_task


def main() -> None:
    crew = Crew(
        agents=[
            blog_researcher,
            blog_writer,
        ],
        tasks=[
            research_task,
            write_task,
        ],
        process=Process.sequential, ## Execute tasks in the order they are listed.
        verbose=True,
        memory=False, ## Do not enable crew-level memory for this introductory version.
        cache=True, ## Allow eligible tool results to be reused from cache.
        max_rpm=10, ## Configure a request-rate limit for the crew.
    )

    result = crew.kickoff(
        inputs={
            ##"topic": "AI vs ML vs DL vs Data Science",
            "topic": "How to deploy AI & LLMs with runpod serverless",
        }
    )

    print("\n========== FINAL BLOG ==========\n")
    print(result)


if __name__ == "__main__":
    main()
