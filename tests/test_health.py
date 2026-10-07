from fastapi.testclient import TestClient

from backend.main import app


def test_health_route_precedes_static_frontend():
    with TestClient(app) as client:
        response = client.get("/health")
        page = client.get("/agent.html")

    assert response.status_code == 200
    assert response.json() == {"status": "ok", "service": "dataagent"}
    assert page.status_code == 200
