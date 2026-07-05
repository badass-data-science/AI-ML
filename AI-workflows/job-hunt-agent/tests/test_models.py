import pytest
from pydantic import ValidationError

from job_hunt_agent.core.models import (
    JobMatchLLMOutput,
    SurfacedBullet,
    SurfacedSkill,
    TokenUsage,
    VariantScore,
)


class TestTokenUsage:
    def test_add_is_immutable(self):
        a = TokenUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15)
        b = TokenUsage(prompt_tokens=1, completion_tokens=1, total_tokens=2)
        c = a + b
        assert c.total_tokens == 17
        # originals untouched
        assert a.total_tokens == 15
        assert b.total_tokens == 2


class TestVariantScore:
    def test_fit_score_bounds(self):
        with pytest.raises(ValidationError):
            VariantScore(
                variant="data-science",
                fit_score=1.5,
                reasoning=["a", "b"],
                strengths=["x"],
            )

    def test_reasoning_min_length_enforced(self):
        with pytest.raises(ValidationError):
            VariantScore(
                variant="data-science",
                fit_score=0.5,
                reasoning=["only one"],
                strengths=["x"],
            )

    def test_unknown_variant_rejected(self):
        with pytest.raises(ValidationError):
            VariantScore(
                variant="not-a-real-variant",
                fit_score=0.5,
                reasoning=["a", "b"],
                strengths=["x"],
            )


class TestJobMatchLLMOutput:
    def _base_kwargs(self, **overrides):
        kwargs = dict(
            variant_scores=[
                VariantScore(
                    variant="data-science", fit_score=0.8, reasoning=["a", "b"], strengths=["x"]
                ),
                VariantScore(
                    variant="bioinformatics", fit_score=0.2, reasoning=["a", "b"], strengths=["x"]
                ),
                VariantScore(
                    variant="ai-engineering", fit_score=0.5, reasoning=["a", "b"], strengths=["x"]
                ),
            ],
            recommended_variant="data-science",
            cover_letter_register="formal-professional",
            register_reasoning="test",
            recommended_achievement_paragraph_ids=["some-id"],
            recommended_soft_skill_id="some-soft-skill",
        )
        kwargs.update(overrides)
        return kwargs

    def test_valid_output_constructs(self):
        out = JobMatchLLMOutput(**self._base_kwargs())
        assert out.recommended_variant == "data-science"

    def test_requires_exactly_three_variant_scores(self):
        kwargs = self._base_kwargs()
        kwargs["variant_scores"] = kwargs["variant_scores"][:2]
        with pytest.raises(ValidationError):
            JobMatchLLMOutput(**kwargs)

    def test_rejects_unknown_recommended_variant(self):
        kwargs = self._base_kwargs(recommended_variant="made-up-variant")
        with pytest.raises(ValidationError):
            JobMatchLLMOutput(**kwargs)

    def test_rejects_unknown_register(self):
        kwargs = self._base_kwargs(cover_letter_register="made-up-register")
        with pytest.raises(ValidationError):
            JobMatchLLMOutput(**kwargs)

    def test_surfaced_skills_and_bullets_optional(self):
        out = JobMatchLLMOutput(**self._base_kwargs())
        assert out.surfaced_skills == []
        assert out.surfaced_bullets == []

    def test_surfaced_skill_and_bullet_construct(self):
        skill = SurfacedSkill(file_title="ML Skills", keyword="TensorFlow", why_relevant="test")
        bullet = SurfacedBullet(bullet_id="x", employer="Acme", why_relevant="test")
        assert skill.keyword == "TensorFlow"
        assert bullet.bullet_id == "x"
