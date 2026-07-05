# AI workflows

A collection of AI workflows:

* **strategic-reports:** An AI workflow that employs LLMs to create strategic reports from recent news articles. By "strategic", I mean that this workflow generates **actionable** business- and career-development recommendations across multiple sectors.
* **job-hunt-agent:** An agentic pipeline that scores a job posting against a decomposed resume/skills knowledge base and assembles a draft resume + cover letter for human review.
* **job-radar:** Pulls job postings from ATS platforms' public APIs (Greenhouse, Lever, Ashby) and scores each for ghost-job risk with deterministic heuristics — feeds `job-hunt-agent` without either project depending on the other.
