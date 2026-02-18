import pathlib

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel

from financial_agent import agent

app = FastAPI(title="Financial AI Agent API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
    message: str


@app.get("/api/health")
def health():
    return {"status": "ok"}


@app.post("/api/chat")
def chat(req: ChatRequest):
    run = agent.run(req.message, stream=False)
    content = run.content if hasattr(run, "content") else str(run)
    return {"response": content}


# Production: serve React build from frontend/dist
DIST = pathlib.Path(__file__).parent / "frontend" / "dist"
if DIST.exists():
    app.mount("/assets", StaticFiles(directory=str(DIST / "assets")), name="assets")

    @app.get("/{full_path:path}", include_in_schema=False)
    def serve_spa(full_path: str):
        return FileResponse(str(DIST / "index.html"))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
