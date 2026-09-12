from fastapi import APIRouter, FastAPI
from fastapi.testclient import TestClient
from prometheus_client import REGISTRY

from sglang.srt.utils.common import add_prometheus_track_response_middleware


router = APIRouter(prefix="/v1")


@router.get("/items/{item_id}")
def item(item_id: str):
    return {"item_id": item_id}


app = FastAPI()
app.include_router(router)
add_prometheus_track_response_middleware(app)

matching_route = next(
    route
    for route in app.routes
    if route.matches(
        {
            "type": "http",
            "method": "GET",
            "path": "/v1/items/42",
            "root_path": "",
        }
    )[0].name
    == "FULL"
)
print("matching_route", type(matching_route).__name__)
print("matching_route_has_path", hasattr(matching_route, "path"))
try:
    matching_route.path
except AttributeError as error:
    print("reported_old_access_exception", repr(error))
else:
    raise AssertionError("Expected the included-router wrapper to lack .path")

response = TestClient(app).get("/v1/items/42")
print("status_code", response.status_code)
print("json", response.json())

request_samples = [
    sample
    for metric in REGISTRY.collect()
    if metric.name == "sglang:http_requests"
    for sample in metric.samples
]
print(
    "request_samples",
    [(sample.name, sample.labels, sample.value) for sample in request_samples],
)

assert response.status_code == 200
assert any(
    sample.labels.get("endpoint") == "/v1/items/42" and sample.value == 1
    for sample in request_samples
)
