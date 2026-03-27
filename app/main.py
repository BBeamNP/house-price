from fastapi import FastAPI
from fastapi.templating import Jinja2Templates
from fastapi.requests import Request
from app.api import deal, roi, goldmine
from fastapi.staticfiles import StaticFiles
app = FastAPI(title="NY Housing ML API")
app.mount("/static", StaticFiles(directory="static"), name="static")


templates = Jinja2Templates(directory="templates")

app.include_router(deal.router)
app.include_router(roi.router)
app.include_router(goldmine.router)

@app.get("/")
def home(request: Request):
    return templates.TemplateResponse(request, "deal.html")


@app.get("/matchmaker")
def matchmaker_page(request: Request):
    return templates.TemplateResponse(request, "matchmaker.html")


@app.get("/goldmine")
def goldmine_map(request: Request):
    return templates.TemplateResponse(request, "goldmine.html")