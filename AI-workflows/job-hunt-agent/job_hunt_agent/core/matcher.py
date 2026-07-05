"""Two-phase job-posting matching: a deterministic keyword prefilter (no LLM,
no network) followed by a single LLM synthesis call scoring all three resume
variants comparatively.

See prompts.py for the prompt construction and job_hunt_agent/core/llm_client.py
for the underlying instructor/litellm client this depends on.
"""

from __future__ import annotations

import re

from job_hunt_agent.core.llm_client import LLMClient
from job_hunt_agent.core.models import JobMatchLLMOutput, JobMatchResult, JobPosting
from job_hunt_agent.core.prompts import build_match_prompt, build_system_prompt
from job_hunt_agent.core.vault_models import ExperienceBullet, SkillFile, VaultSnapshot

_WORD_RE = re.compile(r"[a-z0-9]+")

_STOPWORDS = {
    "a", "an", "the", "and", "or", "but", "of", "to", "in", "on", "for", "with",
    "is", "are", "was", "were", "be", "been", "being", "as", "at", "by", "from",
    "we", "you", "your", "our", "will", "have", "has", "had", "this", "that",
    "these", "those", "it", "its", "not", "who", "what", "which", "into",
    "using", "use", "work", "team", "role", "job", "years", "year", "experience",
}


def _normalize_words(text: str) -> list[str]:
    return _WORD_RE.findall(text.lower())


def _token_set(text: str) -> set[str]:
    words = [w for w in _normalize_words(text) if w not in _STOPWORDS]
    unigrams = set(words)
    bigrams = {f"{a} {b}" for a, b in zip(words, words[1:])}
    return unigrams | bigrams


def _overlap_score(a: set[str], b: set[str]) -> int:
    return len(a & b)


def prefilter_candidates(
    job: JobPosting,
    vault: VaultSnapshot,
    top_n_skills: int = 15,
    top_n_bullets: int = 20,
) -> tuple[list[SkillFile], list[ExperienceBullet]]:
    """Rank vault content by plain keyword overlap with the posting text.

    No LLM call, no network — this exists purely to shrink the vault down to
    a digest small enough to fit comfortably in one prompt, while making sure
    the prompt still contains enough of the "available, not yet used" content
    for the LLM to have a real chance of surfacing a genuine gap.
    """
    posting_tokens = _token_set(job.raw_text)

    scored_skills: list[tuple[int, SkillFile]] = []
    for skill in vault.skills:
        used_keywords = {kw for entry in skill.used_entries for kw in entry.keywords}
        used_score = sum(
            _overlap_score(posting_tokens, _token_set(kw)) for kw in used_keywords
        )
        available_score = sum(
            _overlap_score(posting_tokens, _token_set(kw))
            for kw in skill.available_not_yet_used
        )
        total = used_score + available_score
        if total > 0:
            scored_skills.append((total, skill))
    scored_skills.sort(key=lambda pair: pair[0], reverse=True)
    candidate_skills = [skill for _, skill in scored_skills[:top_n_skills]]

    scored_bullets: list[tuple[int, ExperienceBullet]] = []
    for bullet in vault.all_experience_bullets():
        score = _overlap_score(posting_tokens, _token_set(bullet.text))
        if score > 0:
            scored_bullets.append((score, bullet))
    scored_bullets.sort(key=lambda pair: pair[0], reverse=True)
    candidate_bullets = [bullet for _, bullet in scored_bullets[:top_n_bullets]]

    return candidate_skills, candidate_bullets


async def score_job(
    job: JobPosting,
    vault: VaultSnapshot,
    client: LLMClient,
    top_n_skills: int = 15,
    top_n_bullets: int = 20,
) -> JobMatchResult:
    """Prefilter + one LLM call, error-isolated: never raises, always returns a result."""
    try:
        candidate_skills, candidate_bullets = prefilter_candidates(
            job, vault, top_n_skills=top_n_skills, top_n_bullets=top_n_bullets
        )
        system = build_system_prompt(vault)
        prompt = build_match_prompt(job, vault, candidate_skills, candidate_bullets)

        llm_output, usage = await client.complete_structured(
            prompt=prompt,
            response_model=JobMatchLLMOutput,
            system=system,
        )

        return JobMatchResult(job=job, llm_output=llm_output, token_usage=usage)
    except Exception as exc:  # noqa: BLE001 - deliberate error isolation
        return JobMatchResult(job=job, error=str(exc))
