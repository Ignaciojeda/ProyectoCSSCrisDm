from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
import pickle
import numpy as np
import pandas as pd
import uvicorn

app = FastAPI()

# Directorio de templates
templates = Jinja2Templates(directory="templates")

# Carga de modelos
with open("EquipoGanar.pkl", "rb") as f:
    modelo_clasificacion = pickle.load(f)

with open("KillsPartida.pkl", "rb") as f:
    modelo_regresion = pickle.load(f)

@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse("hub.html", {"request": request})

@app.get("/ganar", response_class=HTMLResponse)
def prediccion_ganar_get(request: Request):
    return templates.TemplateResponse("ganar.html", {"request": request, "resultado": None})

@app.post("/ganar", response_class=HTMLResponse)
def prediccion_ganar_post(
    request: Request,
    RoundStartingEquipmentValue: float = Form(...),
    TeamStartingEquipmentValue: float = Form(...),
    RoundHeadshots: float = Form(...),
    RoundAssists: float = Form(...),
    RoundFlankKills: float = Form(...),
    MatchHeadshots: float = Form(...),
    MatchAssists: float = Form(...),
    MatchFlankKills: float = Form(...),
    RLethalGrenadesThrown: float = Form(...),
):
    input_data = pd.DataFrame([{
        'Avg_RLethalGrenadesThrown': RLethalGrenadesThrown,
        'Sum_RoundKills': 0,  # Puedes ajustar estos valores si tienes más entradas
        'Sum_RoundAssists': RoundAssists,
        'Avg_RoundHeadshots': RoundHeadshots,
        'Avg_RoundFlankKills': RoundFlankKills,
        'Avg_Survived': 1.0,  # Supuesto positivo
        'Avg_RoundStartingEquipmentValue': RoundStartingEquipmentValue,
        'Avg_TeamStartingEquipmentValue': TeamStartingEquipmentValue,
        'Sum_MatchKills': 0,  # Puedes ajustar o estimar según el contexto
        'Sum_MatchFlankKills': MatchFlankKills,
        'Sum_MatchAssists': MatchAssists,
        'Sum_MatchHeadshots': MatchHeadshots,
        'Avg_outlier': 0  # Puedes cambiar si tienes un detector de outliers
    }])

    prediccion = modelo_clasificacion.predict(input_data)[0]

    return templates.TemplateResponse("ganar.html", {
        "request": request,
        "resultado": int(prediccion)
    })

@app.get("/kills", response_class=HTMLResponse)
def prediccion_kills_get(request: Request):
    return templates.TemplateResponse("kills.html", {"request": request, "resultado": None})

@app.post("/kills", response_class=HTMLResponse)
def prediccion_kills_post(
    request: Request,
    Team: str = Form(...),
    RoundStartingEquipmentValue: float = Form(...),
    TeamStartingEquipmentValue: float = Form(...),
    RoundHeadshots: float = Form(...),
    RoundAssists: float = Form(...),
    RoundFlankKills: float = Form(...),
    MatchHeadshots: float = Form(...),
    MatchAssists: float = Form(...),
    MatchFlankKills: float = Form(...),
    RLethalGrenadesThrown: float = Form(...),
):
    input_data = pd.DataFrame([{
        'Team': Team,
        'RoundStartingEquipmentValue': RoundStartingEquipmentValue,
        'TeamStartingEquipmentValue': TeamStartingEquipmentValue,
        'RoundHeadshots': RoundHeadshots,
        'MatchHeadshots': MatchHeadshots,
        'RoundAssists': RoundAssists,
        'RoundFlankKills': RoundFlankKills,
        'MatchAssists': MatchAssists,
        'MatchFlankKills': MatchFlankKills,
        'RLethalGrenadesThrown': RLethalGrenadesThrown,
    }])

    input_dummies = pd.get_dummies(input_data)
    modelo_features = modelo_regresion.feature_names_in_
    for col in modelo_features:
        if col not in input_dummies.columns:
            input_dummies[col] = 0
    input_dummies = input_dummies[modelo_features]

    prediccion = modelo_regresion.predict(input_dummies)[0]

    return templates.TemplateResponse("kills.html", {"request": request, "resultado": prediccion})

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)