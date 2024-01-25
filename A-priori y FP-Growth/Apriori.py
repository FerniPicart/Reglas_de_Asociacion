import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules


ruta_excel = 'C:\\Users\\fxpic\\Downloads\\Facultad\\BDDA\\TP 12 - Reglas de Asociación\\cestaCompras-23.xlsx'

# Excluye la columna de IDs
df = pd.read_excel(ruta_excel, usecols=lambda column: column != 'A')

# Convierte los datos a Bool
df = df.astype(bool)

# Soporte, métricas y umbrales de confianza
# 1
soporteMin_1 = 0.1
metrica_1 = "confidence"
limiteMin_1 = 0.5
# 2
soporteMin_2 = 0.05
metrica_2 = "lift"
limiteMin_2 = 0.7

# Apriori para encontrar conjuntos de ítems frecuentes
conjuntos_frecuentes_ap_1 = apriori(df, min_support=soporteMin_1, use_colnames=True)
conjuntos_frecuentes_ap_2 = apriori(df, min_support=soporteMin_2, use_colnames=True)
# obtiene conjuntos frecuentes
reglas_asociacion_ap_1 = association_rules(conjuntos_frecuentes_ap_1, metric=metrica_1, min_threshold= limiteMin_1)
reglas_asociacion_ap_2 = association_rules(conjuntos_frecuentes_ap_2, metric=metrica_2, min_threshold= limiteMin_2)

# impresion del conjunto 1
print(f"1 - Reglas de Asociación Apriori: con umbral de soporte {soporteMin_1} y métrica {metrica_1} con umbral de confianza {limiteMin_1}:")
print("Cobertura, Soporte y Confianza:")
print(reglas_asociacion_ap_1[['antecedents', 'consequents', 'support', 'confidence']])
print()
print()


# impresion del conjunto 2
print(f"1 - Reglas de Asociación Apriori: con umbral de soporte {soporteMin_2} y métrica {metrica_2} con umbral de confianza {limiteMin_2}:")
print("Cobertura, Soporte y Confianza:")
print(reglas_asociacion_ap_2[['antecedents', 'consequents', 'support', 'confidence']])
print()
