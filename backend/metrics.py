from prometheus_client import Counter, Histogram

# Request count
REQUEST_COUNT = Counter(
    "rag_requests_total",
    "Total number of RAG requests",
    ["endpoint"],
)

# Latency
REQUEST_LATENCY = Histogram(
    "rag_request_latency_seconds",
    "Latency of RAG requests",
    ["endpoint"],
)
