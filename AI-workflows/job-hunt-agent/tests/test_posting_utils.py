from job_hunt_agent.core.posting_utils import extract_company_brief


class TestExtractCompanyBrief:
    def test_about_heading_with_body_in_next_paragraph(self):
        text = """**About Acme Corp**

Acme Corp is committed to turning scientific promise into meaningful innovation for underserved communities around the world.

### Position Summary

The role does things."""
        result = extract_company_brief(text)
        assert result == (
            "Acme Corp is committed to turning scientific promise into "
            "meaningful innovation for underserved communities around the world."
        )

    def test_about_heading_with_body_in_same_paragraph(self):
        text = """**About Acme Corp.** We build things that matter, for people who need them.

### Position Summary

The role does things."""
        result = extract_company_brief(text)
        assert "We build things that matter" in result

    def test_falls_back_to_first_substantial_paragraph_when_no_about_heading(self):
        text = """Acme Corp is a leading provider of widgets, serving customers across many industries with a broad and growing portfolio of products.

### Responsibilities

- Do things
- Do other things"""
        result = extract_company_brief(text)
        assert result is not None
        assert "leading provider of widgets" in result

    def test_short_paragraphs_and_headings_are_skipped_by_fallback(self):
        text = """# Senior Engineer

### Requirements

- Python
- SQL"""
        result = extract_company_brief(text)
        assert result is None

    def test_empty_text_returns_none(self):
        assert extract_company_brief("") is None
