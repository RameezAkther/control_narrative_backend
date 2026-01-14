from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.auth.routes import router as auth_router
from app.users.routes import router as user_router
from app.documents.routes import router as documents_router
from dotenv import load_dotenv
import os

load_dotenv()
print(os.getenv("GOOGLE_API_KEY"))


app = FastAPI()

# ---------------------------
# ENABLE CORS FOR FRONTEND
# ---------------------------
origins = [
    "http://localhost:5173",   # Vite dev server
    "http://127.0.0.1:5173",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],     # allow GET, POST, DELETE, OPTIONS, PUT...
    allow_headers=["*"],
)

# ROUTES
app.include_router(auth_router)
app.include_router(user_router)
app.include_router(documents_router)

@app.get("/")
def home():
    return {"message": "Backend running locally!"}
