from fastapi import FastAPI
from fastapi.templating import Jinja2Templates
from fastapi.requests import Request
from app.api import deal, roi, goldmine

app = FastAPI(title="NY Housing ML API")

templates = Jinja2Templates(directory="templates")

app.include_router(deal.router)
app.include_router(roi.router)
app.include_router(goldmine.router)

@app.get("/")
def home(request: Request):
    return templates.TemplateResponse("deal.html", {"request": request})
