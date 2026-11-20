# app.py
# FastAPI backend. Run it with:  uvicorn app:app --reload
# Then open http://127.0.0.1:8000 in the browser.

from fastapi import FastAPI
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

import predictor

app = FastAPI(title="Crop Recommendation with Confidence and Out-of-Range Detection")

limits = predictor.VALID_LIMITS


# The data we expect from the frontend. Pydantic checks the type and range for us.
class CropInput(BaseModel):
    N: float = Field(ge=limits["N"][0], le=limits["N"][1], allow_inf_nan=False)
    P: float = Field(ge=limits["P"][0], le=limits["P"][1], allow_inf_nan=False)
    K: float = Field(ge=limits["K"][0], le=limits["K"][1], allow_inf_nan=False)
    temperature: float = Field(ge=limits["temperature"][0], le=limits["temperature"][1], allow_inf_nan=False)
    humidity: float = Field(ge=limits["humidity"][0], le=limits["humidity"][1], allow_inf_nan=False)
    ph: float = Field(ge=limits["ph"][0], le=limits["ph"][1], allow_inf_nan=False)
    rainfall: float = Field(ge=limits["rainfall"][0], le=limits["rainfall"][1], allow_inf_nan=False)


# If the input is wrong (for example the word "hello"), send back a simple message
# for each wrong field instead of the long default error.
@app.exception_handler(RequestValidationError)
def handle_bad_input(request, error):
    errors = {}
    for item in error.errors():
        field = str(item["loc"][-1])
        errors[field] = "Invalid input"
    return JSONResponse(status_code=422, content={"errors": errors})


@app.post("/predict")
def predict(data: CropInput):
    values = data.model_dump()
    result = predictor.make_recommendation(values)
    return result


@app.get("/health")
def health():
    return {"status": "ok", "model": predictor.bundle["model_name"]}


@app.get("/")
def home():
    return FileResponse("static/index.html")


# serves style.css, validation.js and script.js
app.mount("/static", StaticFiles(directory="static"), name="static")
