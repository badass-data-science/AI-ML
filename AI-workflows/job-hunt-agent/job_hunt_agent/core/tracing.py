"""Optional LLM observability integrations — both opt-in via environment variables.

Ported from strategic-reports/strategic_reports/daily/core/tracing.py, unchanged
in mechanism. Neither backend raises on misconfiguration — tracing is
observability infrastructure, not business logic, and should never be the
reason a match/draft run fails.
"""

import os
import uuid

import litellm
import structlog

log = structlog.get_logger(__name__)


def generate_run_id() -> str:
    """Return a UUID string to identify one pipeline run across all its LLM calls."""
    return str(uuid.uuid4())


def setup_langfuse() -> bool:
    """Enable litellm's Langfuse callback if credentials are present."""
    if not (os.environ.get("LANGFUSE_PUBLIC_KEY") and os.environ.get("LANGFUSE_SECRET_KEY")):
        log.debug("langfuse_skipped", reason="LANGFUSE_PUBLIC_KEY / LANGFUSE_SECRET_KEY not set")
        return False

    try:
        existing_success = list(getattr(litellm, "success_callback", []) or [])
        existing_failure = list(getattr(litellm, "failure_callback", []) or [])

        if "langfuse" not in existing_success:
            litellm.success_callback = existing_success + ["langfuse"]
        if "langfuse" not in existing_failure:
            litellm.failure_callback = existing_failure + ["langfuse"]

        host = os.environ.get("LANGFUSE_HOST", "https://cloud.langfuse.com")
        log.info("langfuse_enabled", host=host)
        return True

    except Exception as exc:
        log.warning("langfuse_setup_failed", error=str(exc))
        return False


def setup_phoenix() -> bool:
    """Start a local Phoenix server and auto-instrument litellm via OpenTelemetry."""
    if not os.environ.get("PHOENIX_TRACING"):
        log.debug("phoenix_skipped", reason="PHOENIX_TRACING not set")
        return False

    try:
        import phoenix as px
        from openinference.instrumentation.litellm import LiteLLMInstrumentor
        from opentelemetry import trace
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import SimpleSpanProcessor

        session = px.launch_app()
        endpoint = session.url.rstrip("/") + "/v1/traces"

        provider = TracerProvider()
        provider.add_span_processor(SimpleSpanProcessor(OTLPSpanExporter(endpoint=endpoint)))
        trace.set_tracer_provider(provider)

        LiteLLMInstrumentor().instrument()

        log.info("phoenix_enabled", ui=session.url)
        return True

    except ImportError:
        log.warning(
            "phoenix_setup_failed",
            reason="Required packages not installed",
            hint="pip install arize-phoenix openinference-instrumentation-litellm",
        )
        return False
    except Exception as exc:
        log.warning("phoenix_setup_failed", error=str(exc))
        return False


def setup_tracing() -> dict[str, bool]:
    """Configure all available tracing backends. Returns which backends are active."""
    return {
        "langfuse": setup_langfuse(),
        "phoenix": setup_phoenix(),
    }
