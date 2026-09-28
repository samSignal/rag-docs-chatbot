import importlib

from fastapi.testclient import TestClient


def make_client(tmp_path, monkeypatch):
    monkeypatch.setenv("INDEX_DIR", str(tmp_path))
    monkeypatch.setenv("LLM_PROVIDER", "offline")
    monkeypatch.setenv("EMBED_PROVIDER", "offline")
    import app.main

    importlib.reload(app.main)
    return TestClient(app.main.app)


def test_upload_ask_delete_flow(tmp_path, monkeypatch):
    c = make_client(tmp_path, monkeypatch)
    assert c.get("/health").json()["chunks"] == 0

    doc = b"The office wifi password is changed every Monday. Guests use the network called Visitor."
    r = c.post("/documents", files={"file": ("office.txt", doc, "text/plain")})
    assert r.status_code == 200 and r.json()["chunks"] == 1

    r = c.post("/ask", json={"question": "Which network do guests use?"})
    body = r.json()
    assert r.status_code == 200
    assert "Visitor" in body["answer"]
    assert body["sources"][0]["source"] == "office.txt"

    assert c.get("/documents").json()["documents"] == [{"name": "office.txt", "chunks": 1}]
    assert c.delete("/documents/office.txt").status_code == 200
    assert c.delete("/documents/office.txt").status_code == 404


def test_rejects_bad_input(tmp_path, monkeypatch):
    c = make_client(tmp_path, monkeypatch)
    assert c.post("/documents", files={"file": ("x.png", b"123", "image/png")}).status_code == 400
    assert c.post("/ask", json={"question": ""}).status_code == 422
    assert c.get("/").status_code == 200
