from fastapi import FastAPI, Request, Response
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST

from backend.routes import router
from backend.metrics import REQUEST_COUNT, REQUEST_LATENCY

app = FastAPI(
    title="Ultimate RAG Platform",
    description="Production-grade Retrieval-Augmented Generation system",
    version="1.0.0",
)

@app.middleware("http")
async def metrics_middleware(request: Request, call_next):
    endpoint = request.url.path

    with REQUEST_LATENCY.labels(endpoint=endpoint).time():
        response = await call_next(request)

    REQUEST_COUNT.labels(endpoint=endpoint).inc()
    return response


@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


app.include_router(router)
