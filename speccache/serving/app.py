"""FastAPI Server-Sent Events (SSE) streaming server with custom hardware telemetry headers.

Exposes:
- POST /generate and POST /v1/chat/completions (SSE streaming and standard JSON).
- Custom hardware telemetry headers:
  * x-speccache-route: DIRECT_REUSE | SPECULATIVE_VERIFICATION | COLD_MISS
  * x-speccache-similarity: Cosine similarity score
  * x-speccache-ttft-ms: Time To First Token in ms
  * x-speccache-tokens-per-sec: Generation throughput
  * x-speccache-spec-accepted-tokens: Accepted speculative draft tokens count
  * x-speccache-dma-bytes-saved: Bytes of prefill saved via INT8 DMA
  * x-speccache-kv-blocks-reused: Reused cache blocks count
- GET /health: Pool capacity and system health.
- GET /metrics: Prometheus-compatible telemetry metrics.
"""

import json
from typing import Any, AsyncIterator, Dict, List, Optional, Union
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

from speccache.cache.engine import SpecCacheEngine
from speccache.serving.pipeline import GenerationResult, RoutePath, SpecCacheLLMRunner


class GenerateRequest(BaseModel):
    """Payload for generation endpoints."""
    prompt: Optional[str] = None
    tokens: Optional[List[int]] = None
    max_tokens: int = Field(default=32, ge=1, le=2048)
    temperature: float = Field(default=0.0, ge=0.0)
    stream: bool = True


class ChatMessage(BaseModel):
    """Individual chat message."""
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    """OpenAI-compatible chat completion payload."""
    messages: Optional[List[ChatMessage]] = None
    prompt: Optional[str] = None
    tokens: Optional[List[int]] = None
    max_tokens: int = Field(default=32, ge=1, le=2048)
    temperature: float = Field(default=0.0, ge=0.0)
    stream: bool = True


def _format_sse_event(data: Dict[str, Any]) -> str:
    """Format dictionary into standard Server-Sent Event data block."""
    return f"data: {json.dumps(data)}\n\n"


def create_app(runner: Optional[SpecCacheLLMRunner] = None) -> FastAPI:
    """Create and configure FastAPI application for SpecCache inference serving.

    Args:
        runner: Pre-configured SpecCacheLLMRunner. If None, default instance is built.

    Returns:
        FastAPI app instance.
    """
    app = FastAPI(
        title="SpecCache Inference Engine",
        version="0.3.0",
        description="High-Throughput Semantic KV-Cache Engine with Speculative Draft Verification",
    )

    # State
    if runner is None:
        engine = SpecCacheEngine(
            chunk_size=64,
            num_kv_heads=4,
            head_dim=64,
            max_blocks=64,
            embedding_dim=64,
        )
        runner = SpecCacheLLMRunner(engine=engine)

    app.state.runner = runner
    app.state.metrics = {
        "requests_total": 0,
        "direct_reuse_total": 0,
        "spec_verification_total": 0,
        "cold_miss_total": 0,
        "dma_bytes_saved_total": 0,
        "spec_accepted_total": 0,
        "ttft_ms_sum": 0.0,
    }

    def _resolve_tokens(req: Union[GenerateRequest, ChatCompletionRequest]) -> List[int]:
        if req.tokens is not None and len(req.tokens) > 0:
            return req.tokens
        text = req.prompt or ""
        if isinstance(req, ChatCompletionRequest) and req.messages:
            text = " ".join(m.content for m in req.messages)
        if not text:
            # Default placeholder tokens if empty
            return list(range(100, 100 + app.state.runner.chunk_size))
        # Hash text characters to synthetic token IDs if tokenizer is absent
        if app.state.runner.tokenizer is not None:
            return app.state.runner.tokenizer.encode(text)
        return [ord(c) % 32000 for c in text]

    def _build_telemetry_headers(res: GenerationResult) -> Dict[str, str]:
        return {
            "x-speccache-route": res.route.value,
            "x-speccache-similarity": f"{res.similarity:.4f}",
            "x-speccache-ttft-ms": f"{res.ttft_ms:.2f}",
            "x-speccache-tokens-per-sec": f"{res.decode_tokens_per_sec:.2f}",
            "x-speccache-spec-accepted-tokens": str(res.spec_accepted_count),
            "x-speccache-dma-bytes-saved": str(res.dma_bytes_saved),
            "x-speccache-kv-blocks-reused": str(res.kv_blocks_reused),
        }

    def _record_metrics(res: GenerationResult) -> None:
        metrics = app.state.metrics
        metrics["requests_total"] += 1
        metrics["dma_bytes_saved_total"] += res.dma_bytes_saved
        metrics["spec_accepted_total"] += res.spec_accepted_count
        metrics["ttft_ms_sum"] += res.ttft_ms
        if res.route == RoutePath.DIRECT_REUSE:
            metrics["direct_reuse_total"] += 1
        elif res.route == RoutePath.SPECULATIVE_VERIFICATION:
            metrics["spec_verification_total"] += 1
        else:
            metrics["cold_miss_total"] += 1

    @app.get("/health")
    def health() -> Dict[str, Any]:
        """Health and KV pool slab capacity status."""
        active_runner: SpecCacheLLMRunner = app.state.runner
        return {
            "status": "ok",
            "phase": 3,
            "pool_capacity": active_runner.engine.pool.max_blocks,
            "allocated_blocks": active_runner.engine.pool.allocated_count,
            "free_blocks": active_runner.engine.pool.free_count,
            "chunk_size": active_runner.chunk_size,
            "draft_gamma": active_runner.draft_gamma,
        }

    @app.get("/metrics")
    def metrics() -> Response:
        """Prometheus-compatible hardware telemetry metrics."""
        m = app.state.metrics
        reqs = max(1, m["requests_total"])
        avg_ttft = m["ttft_ms_sum"] / reqs if m["requests_total"] > 0 else 0.0

        body = (
            f"# HELP speccache_requests_total Total number of inference requests processed\n"
            f"# TYPE speccache_requests_total counter\n"
            f"speccache_requests_total {m['requests_total']}\n"
            f"speccache_requests_by_route{{route=\"DIRECT_REUSE\"}} {m['direct_reuse_total']}\n"
            f"speccache_requests_by_route{{route=\"SPECULATIVE_VERIFICATION\"}} {m['spec_verification_total']}\n"
            f"speccache_requests_by_route{{route=\"COLD_MISS\"}} {m['cold_miss_total']}\n\n"
            f"# HELP speccache_dma_bytes_saved_total Total prefill bytes eliminated via INT8 DMA\n"
            f"# TYPE speccache_dma_bytes_saved_total counter\n"
            f"speccache_dma_bytes_saved_total {m['dma_bytes_saved_total']}\n\n"
            f"# HELP speccache_spec_accepted_total Total speculative draft tokens accepted\n"
            f"# TYPE speccache_spec_accepted_total counter\n"
            f"speccache_spec_accepted_total {m['spec_accepted_total']}\n\n"
            f"# HELP speccache_ttft_ms_avg Average Time-To-First-Token in milliseconds\n"
            f"# TYPE speccache_ttft_ms_avg gauge\n"
            f"speccache_ttft_ms_avg {avg_ttft:.2f}\n"
        )
        return Response(content=body, media_type="text/plain")

    @app.post("/generate")
    async def generate_endpoint(req: GenerateRequest) -> Response:
        """Generate tokens with optional SSE streaming."""
        tokens = _resolve_tokens(req)
        active_runner: SpecCacheLLMRunner = app.state.runner

        if not req.stream:
            result = active_runner.generate(
                prompt_tokens=tokens,
                max_new_tokens=req.max_tokens,
                temperature=req.temperature,
            )
            _record_metrics(result)
            headers = _build_telemetry_headers(result)
            return JSONResponse(
                content={
                    "tokens": result.generated_tokens,
                    "text": result.text,
                    "route": result.route.value,
                    "similarity": result.similarity,
                    "ttft_ms": result.ttft_ms,
                    "tokens_per_sec": result.decode_tokens_per_sec,
                    "spec_accepted_count": result.spec_accepted_count,
                    "dma_bytes_saved": result.dma_bytes_saved,
                },
                headers=headers,
            )

        # Streaming mode: execute generation and stream chunks via SSE
        result = active_runner.generate(
            prompt_tokens=tokens,
            max_new_tokens=req.max_tokens,
            temperature=req.temperature,
        )
        _record_metrics(result)
        headers = _build_telemetry_headers(result)

        async def sse_event_stream() -> AsyncIterator[str]:
            for i, tok in enumerate(result.generated_tokens):
                finish = "stop" if i == len(result.generated_tokens) - 1 else None
                tok_text = str(tok)
                if active_runner.tokenizer is not None:
                    tok_text = active_runner.tokenizer.decode([tok])

                event_data = {
                    "token": tok_text,
                    "token_id": tok,
                    "index": i,
                    "finish_reason": finish,
                }
                yield _format_sse_event(event_data)
            yield "data: [DONE]\n\n"

        return StreamingResponse(
            sse_event_stream(),
            media_type="text/event-stream",
            headers=headers,
        )

    @app.post("/v1/chat/completions")
    async def chat_completions(req: ChatCompletionRequest) -> Response:
        """OpenAI-compatible chat completions endpoint."""
        gen_req = GenerateRequest(
            prompt=req.prompt,
            tokens=req.tokens,
            max_tokens=req.max_tokens,
            temperature=req.temperature,
            stream=req.stream,
        )
        if req.messages and not gen_req.prompt:
            gen_req.prompt = " ".join(m.content for m in req.messages)

        return await generate_endpoint(gen_req)

    return app


# Default application instance
app = create_app()
