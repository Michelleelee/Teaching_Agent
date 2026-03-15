from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from qbank_agent.api.routes import student, admin
from qbank_agent import config

app = FastAPI(
    title="QBank-Agent API",
    description="FastAPI Backend for the Educational Agent System.",
    version="1.0.0"
)

# Allow CORS for local development / frontend integrations
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Change this in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Startup event to ensure directories exist
@app.on_event("startup")
async def startup_event():
    config.setup_directories()

# Mount routers
app.include_router(student.router, prefix="/api/v1/student", tags=["Student"])
app.include_router(admin.router, prefix="/api/v1/admin", tags=["Admin"])

@app.get("/")
def read_root():
    return {"message": "Welcome to the QBank-Agent API. Check /docs for endpoints."}
