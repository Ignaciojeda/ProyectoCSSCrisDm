from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
import pickle
import numpy as np
import pandas as pd
import uvicorn
import os

app = FastAPI()



templates = Jinja2Templates(directory="templates")

# Carga de modelos
with open("EquipoGanar.pkl", "rb") as f:
    modelo_clasificacion = pickle.load(f)

with open("KillsPartida.pkl", "rb") as f:
    modelo_regresion = pickle.load(f)

@app.get("/test_static")
def test_static():
    return {"css_url": app.url_path_for("static", filename="estilo.css")}

@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse("hub.html", {"request": request})

@app.get("/ganar", response_class=HTMLResponse)
def prediccion_ganar_get(request: Request):
    return templates.TemplateResponse("ganar.html", {"request": request, "resultado": None})

@app.post("/ganar", response_class=HTMLResponse)
def prediccion_ganar_post(
    request: Request,
    Team: str = Form(...),
    RoundStartingEquipmentValue: float = Form(...),
    TeamStartingEquipmentValue: float = Form(...),
    MatchHeadshots: float = Form(...),
    MatchAssists: float = Form(...),
    MatchFlankKills: float = Form(...),
    RLethalGrenadesThrown: float = Form(...),
    Map: str = Form(...)
):
    input_data = pd.DataFrame([{
        'Team': Team,
        'RoundStartingEquipmentValue': RoundStartingEquipmentValue,
        'TeamStartingEquipmentValue': TeamStartingEquipmentValue,
        'MatchHeadshots': MatchHeadshots,
        'MatchAssists': MatchAssists,
        'MatchFlankKills': MatchFlankKills,
        'RLethalGrenadesThrown': RLethalGrenadesThrown,
        'Map': Map
    }])

    input_dummies = pd.get_dummies(input_data)

    modelo_features = modelo_clasificacion.feature_names_in_
    for col in modelo_features:
        if col not in input_dummies.columns:
            input_dummies[col] = 0

    input_dummies = input_dummies[modelo_features]

    prediccion = modelo_clasificacion.predict(input_dummies)[0]

    return templates.TemplateResponse("ganar.html", {"request": request, "resultado": prediccion})


@app.get("/kills", response_class=HTMLResponse)
def prediccion_kills_get(request: Request):
    return templates.TemplateResponse("kills.html", {"request": request, "resultado": None})

@app.post("/kills", response_class=HTMLResponse)
def prediccion_kills_post(
    request: Request,
    Team: str = Form(...),
    RoundStartingEquipmentValue: float = Form(...),
    TeamStartingEquipmentValue: float = Form(...),
    MatchHeadshots: float = Form(...),
    MatchAssists: float = Form(...),
    MatchFlankKills: float = Form(...),
    RLethalGrenadesThrown: float = Form(...),
):
    # Crear DataFrame con los datos de entrada
    input_data = pd.DataFrame([{
        'Team': Team,
        'RoundStartingEquipmentValue': RoundStartingEquipmentValue,
        'TeamStartingEquipmentValue': TeamStartingEquipmentValue,
        'MatchHeadshots': MatchHeadshots,
        'MatchAssists': MatchAssists,
        'MatchFlankKills': MatchFlankKills,
        'RLethalGrenadesThrown': RLethalGrenadesThrown,
    }])

    # Convertir las variables categóricas a dummies
    input_dummies = pd.get_dummies(input_data)

    # Alinear columnas con las del modelo entrenado
    # (esto es necesario si el modelo fue entrenado con dummies)
    modelo_features = modelo_regresion.feature_names_in_
    for col in modelo_features:
        if col not in input_dummies.columns:
            input_dummies[col] = 0  # Agregar columnas faltantes con 0

    input_dummies = input_dummies[modelo_features]

    # Hacer la predicción
    prediccion = modelo_regresion.predict(input_dummies)[0]

    return templates.TemplateResponse("kills.html", {"request": request, "resultado": prediccion})

