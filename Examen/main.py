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
    Avg_RLethalGrenadesThrown: float = Form(...),
    Sum_RoundKills: float = Form(...),
    Sum_RoundAssists: float = Form(...),
    Avg_RoundHeadshots: float = Form(...),
    Avg_RoundFlankKills: float = Form(...),
    Avg_Survived: float = Form(...),
    Avg_RoundStartingEquipmentValue: float = Form(...),
    Avg_TeamStartingEquipmentValue: float = Form(...),
    Sum_MatchKills: float = Form(...),
    Sum_MatchFlankKills: float = Form(...),
    Sum_MatchAssists: float = Form(...),
    Sum_MatchHeadshots: float = Form(...),
    Avg_outlier: float = Form(...),
):
    input_data = pd.DataFrame([{
        'Avg_RLethalGrenadesThrown': Avg_RLethalGrenadesThrown,
        'Sum_RoundKills': Sum_RoundKills,
        'Sum_RoundAssists': Sum_RoundAssists,
        'Avg_RoundHeadshots': Avg_RoundHeadshots,
        'Avg_RoundFlankKills': Avg_RoundFlankKills,
        'Avg_Survived': Avg_Survived,
        'Avg_RoundStartingEquipmentValue': Avg_RoundStartingEquipmentValue,
        'Avg_TeamStartingEquipmentValue': Avg_TeamStartingEquipmentValue,
        'Sum_MatchKills': Sum_MatchKills,
        'Sum_MatchFlankKills': Sum_MatchFlankKills,
        'Sum_MatchAssists': Sum_MatchAssists,
        'Sum_MatchHeadshots': Sum_MatchHeadshots,
        'Avg_outlier': Avg_outlier,
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