from pathlib import Path
from unittest.mock import MagicMock

from job_hunt_agent.core.llm_client import LLMClient
from job_hunt_agent.core.matcher import prefilter_candidates, score_job
from job_hunt_agent.core.models import JobMatchLLMOutput, JobPosting, TokenUsage
from job_hunt_agent.core.vault_reader import load_vault


def make_mock_client(
    llm_output: JobMatchLLMOutput | None = None,
    usage: TokenUsage | None = None,
    raises: Exception | None = None,
) -> MagicMock:
    """Same technique as strategic-reports' make_mock_client: a hand-written
    async dispatch function assigned onto a MagicMock(spec=LLMClient), since a
    generic AsyncMock can't express "raise on this call" vs "return this typed
    value" as cleanly.
    """
    usage = usage or TokenUsage(total_tokens=42)

    async def _complete_structured(prompt, response_model, system=None):
        if raises:
            raise raises
        assert response_model is JobMatchLLMOutput
        return llm_output, usage

    client = MagicMock(spec=LLMClient)
    client.complete_structured = _complete_structured
    return client


class TestPrefilterCandidates:
    def test_ranks_skills_by_keyword_overlap(self, fixture_vault: Path):
        vault = load_vault(fixture_vault)
        job = JobPosting(
            raw_text="Looking for a Data Scientist with strong Python and scikit-learn skills."
        )
        candidate_skills, _ = prefilter_candidates(job, vault)
        titles = [s.title for s in candidate_skills]
        assert "ML Skills" in titles
        # Bio Skills has zero keyword overlap with this posting text
        assert "Bio Skills" not in titles

    def test_ranks_bullets_by_keyword_overlap(self, fixture_vault: Path):
        vault = load_vault(fixture_vault)
        job = JobPosting(raw_text="We need genomics and CRISPR variant calling experience.")
        _, candidate_bullets = prefilter_candidates(job, vault)
        ids = [b.bullet_id for b in candidate_bullets]
        assert "acme-genomics" in ids

    def test_no_overlap_yields_no_candidates(self, fixture_vault: Path):
        vault = load_vault(fixture_vault)
        job = JobPosting(raw_text="zzz qqq xxx completely unrelated nonsense terms")
        candidate_skills, candidate_bullets = prefilter_candidates(job, vault)
        assert candidate_skills == []
        assert candidate_bullets == []

    def test_respects_top_n_limits(self, fixture_vault: Path):
        vault = load_vault(fixture_vault)
        job = JobPosting(
            raw_text="Python machine learning scikit-learn genomics CRISPR variant calling"
        )
        candidate_skills, candidate_bullets = prefilter_candidates(
            job, vault, top_n_skills=1, top_n_bullets=1
        )
        assert len(candidate_skills) <= 1
        assert len(candidate_bullets) <= 1


class TestScoreJob:
    async def test_happy_path_returns_llm_output(
        self, fixture_vault: Path, sample_job_posting, sample_llm_output
    ):
        vault = load_vault(fixture_vault)
        client = make_mock_client(llm_output=sample_llm_output)

        result = await score_job(sample_job_posting, vault, client)

        assert result.error is None
        assert result.llm_output is sample_llm_output
        assert result.token_usage.total_tokens == 42

    async def test_llm_exception_is_isolated_not_raised(
        self, fixture_vault: Path, sample_job_posting
    ):
        vault = load_vault(fixture_vault)
        client = make_mock_client(raises=RuntimeError("rate limited"))

        result = await score_job(sample_job_posting, vault, client)

        assert result.error is not None
        assert "rate limited" in result.error
        assert result.llm_output is None

    async def test_prefilter_exception_is_also_isolated(
        self, fixture_vault: Path, sample_job_posting, monkeypatch
    ):
        vault = load_vault(fixture_vault)
        client = make_mock_client()

        def _boom(*args, **kwargs):
            raise ValueError("prefilter broke")

        monkeypatch.setattr("job_hunt_agent.core.matcher.prefilter_candidates", _boom)

        result = await score_job(sample_job_posting, vault, client)

        assert result.error is not None
        assert "prefilter broke" in result.error
