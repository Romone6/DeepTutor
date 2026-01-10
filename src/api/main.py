from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from src.api.routers import (
    agent_config,
    chat,
    co_writer,
    dashboard,
    debug,
    embedding_provider,
    exam_simulator,
    guide,
    ideagen,
    knowledge,
    llm_provider,
    memory,
    metrics,
    notebook,
    question,
    research,
    settings,
    solve,
    sprint,
    system,
    voice,
)
from src.logging import get_logger

from src.api.middleware import add_trace_middleware

logger = get_logger("API")

app = FastAPI(title="DeepTutor API", version="1.0.0")

add_trace_middleware(app)


@app.on_event("startup")
async def startup_event():
    logger.info("Application startup")
    # Run startup health check
    try:
        from src.services.degradation import get_degradation_service

        service = get_degradation_service()
        import asyncio

        results = await service.run_health_check()
        healthy_count = sum(1 for s in results.values() if s.healthy)
        total_count = len(results)
        logger.info(f"Health check: {healthy_count}/{total_count} components healthy")
        if healthy_count < total_count:
            degraded = [c.value for c, s in results.items() if not s.healthy]
            logger.warning(f"Degraded components: {', '.join(degraded)}")
        service.start_background_monitoring()
    except Exception as e:
        logger.warning(f"Health check skipped: {e}")


# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount user directory as static root for generated artifacts
# This allows frontend to access generated artifacts (images, PDFs, etc.)
# URL: /api/outputs/solve/solve_xxx/artifacts/image.png
# Physical Path: DeepTutor/data/user/solve/solve_xxx/artifacts/image.png
project_root = Path(__file__).parent.parent.parent
user_dir = project_root / "data" / "user"

# Initialize user directories on startup
try:
    from src.services.setup import init_user_directories

    init_user_directories(project_root)
except Exception:
    # Fallback: just create the main directory if it doesn't exist
    if not user_dir.exists():
        user_dir.mkdir(parents=True)

app.mount("/api/outputs", StaticFiles(directory=str(user_dir)), name="outputs")

# Include routers
app.include_router(solve.router, prefix="/api/v1", tags=["solve"])
app.include_router(chat.router, prefix="/api/v1", tags=["chat"])
app.include_router(question.router, prefix="/api/v1/question", tags=["question"])
app.include_router(research.router, prefix="/api/v1/research", tags=["research"])
app.include_router(knowledge.router, prefix="/api/v1/knowledge", tags=["knowledge"])
app.include_router(dashboard.router, prefix="/api/v1/dashboard", tags=["dashboard"])
app.include_router(co_writer.router, prefix="/api/v1/co_writer", tags=["co_writer"])
app.include_router(notebook.router, prefix="/api/v1/notebook", tags=["notebook"])
app.include_router(guide.router, prefix="/api/v1/guide", tags=["guide"])
app.include_router(ideagen.router, prefix="/api/v1/ideagen", tags=["ideagen"])
app.include_router(settings.router, prefix="/api/v1/settings", tags=["settings"])
app.include_router(system.router, prefix="/api/v1/system", tags=["system"])
app.include_router(llm_provider.router, prefix="/api/v1/config/llm", tags=["config"])
app.include_router(embedding_provider.router, prefix="/api/v1/config/embedding", tags=["config"])
app.include_router(agent_config.router, prefix="/api/v1/config", tags=["config"])
app.include_router(voice.router, prefix="/api/v1", tags=["voice"])
app.include_router(memory.router, prefix="/api/v1/memory", tags=["memory"])
app.include_router(sprint.router, prefix="/api/v1/sprint", tags=["sprint"])
app.include_router(metrics.router, prefix="/api/v1/dashboard", tags=["dashboard"])
app.include_router(debug.router, prefix="/api/v1", tags=["debug"])
app.include_router(exam_simulator.router, prefix="/api/v1/exam", tags=["exam"])


@app.get("/")
async def root():
    return {"message": "Welcome to DeepTutor API"}


@app.get("/api/v1/health")
async def health_check():
    """
    Simple health check endpoint for connection testing.
    Returns basic status for monitoring and client connection tests.
    """
    from datetime import datetime

    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "version": "1.0.0",
    }


if __name__ == "__main__":
    from pathlib import Path

    import uvicorn

    # Get project root directory
    project_root = Path(__file__).parent.parent.parent

    # Ensure project root is in Python path
    import sys

    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    # Get port from configuration
    from src.services.setup import get_backend_port

    backend_port = get_backend_port(project_root)

    # Configure reload_excludes with absolute paths to properly exclude directories
    venv_dir = project_root / "venv"
    data_dir = project_root / "data"
    reload_excludes = [
        str(d)
        for d in [
            venv_dir,
            project_root / ".venv",
            data_dir,
            project_root / "web" / "node_modules",
            project_root / "web" / ".next",
            project_root / ".git",
        ]
        if d.exists()
    ]

    uvicorn.run(
        "api.main:app",
        host="0.0.0.0",
        port=backend_port,
        reload=True,
        reload_excludes=reload_excludes,
    )
