from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from src.inference import load_legal_simplifier

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# Load model at startup
pipe = load_legal_simplifier()

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/simplify")
async def simplify(request: Request):
    form_data = await request.form()
    legal_clause = form_data["clause"]
    simplified = simplify_clause(pipe, legal_clause)
    return {"original": legal_clause, "simplified": simplified}