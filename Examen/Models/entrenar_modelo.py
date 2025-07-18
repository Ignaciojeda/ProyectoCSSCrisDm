import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import pickle

data = pd.read_csv("../Dataset/datos_datacss.csv")

# Agregación primero
df_agg = data.groupby(['MatchId', 'InternalTeamId', 'MatchResult']).agg(
    Avg_RLethalGrenadesThrown=('RLethalGrenadesThrown', 'mean'),
    Sum_RoundKills=('RoundKills', 'sum'),
    Sum_RoundAssists=('RoundAssists', 'sum'),
    Avg_RoundHeadshots=('RoundHeadshots', 'mean'),
    Avg_RoundFlankKills=('RoundFlankKills', 'mean'),
    Avg_Survived=('Survived', 'mean'),
    Avg_RoundStartingEquipmentValue=('RoundStartingEquipmentValue', 'mean'),
    Avg_TeamStartingEquipmentValue=('TeamStartingEquipmentValue', 'mean'),
    Sum_MatchKills=('MatchKills', 'first'),
    Sum_MatchFlankKills=('MatchFlankKills', 'first'),
    Sum_MatchAssists=('MatchAssists', 'first'),
    Sum_MatchHeadshots=('MatchHeadshots', 'first'),
    Avg_outlier=('outlier', 'mean')
).reset_index()

features_agg = [
    'Avg_RLethalGrenadesThrown',
    'Sum_RoundKills',
    'Sum_RoundAssists',
    'Avg_RoundHeadshots',
    'Avg_RoundFlankKills',
    'Avg_Survived',
    'Avg_RoundStartingEquipmentValue',
    'Avg_TeamStartingEquipmentValue',
    'Sum_MatchKills',
    'Sum_MatchFlankKills',
    'Sum_MatchAssists',
    'Sum_MatchHeadshots',
    'Avg_outlier'
]

# Escalar las columnas agregadas
scaler = StandardScaler()
df_agg[features_agg] = scaler.fit_transform(df_agg[features_agg])

X = df_agg[features_agg]
y = df_agg['MatchResult']

model = RandomForestClassifier(
    n_estimators=500,
    max_depth=10,
    class_weight='balanced',
    max_features='sqrt',
    random_state=42
)
model.fit(X, y)

pickle.dump(model, open("modelo_rf_clasificacion.pkl", "wb"))
pickle.dump(scaler, open("scaler_rf_clasificacion.pkl", "wb"))

print("✅ Modelo y scaler guardados correctamente.")
