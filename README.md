Voici un README clair et opérationnel.

# Projet — Prévision de la demande 2025, analyse spatio-temporelle, assistant LLM

## Objectif

Prévoir la demande mensuelle 2025 par produit et localisation, visualiser les hotspots géographiques, analyser tendances saisonnières et lancer un assistant LLM orienté décisions logistiques et d’approvisionnement.

## Structure

```
.
├── dataset/
│   └── all_data.csv
├── notebook/
│   └── notebook.ipynb
├── outputs/
│   ├── city_month_matrix_2025_AAA_Batteries_(4-pack).csv
│   ├── city_month_matrix_2025_ThinkPad_Laptop.csv
│   ├── geo_trend_volatility_2025.csv
│   ├── predictions_2025_aaa_batteries_4_pack.csv
│   ├── predictions_2025_thinkpad_laptop.csv
│   └── timeseries_2025_global.csv
├── assistant-LLM/
│   ├── main.py
│   └── requirements.txt  # gelé via pip freeze
└── docker/
    └── Dockerfile
```

## Pipeline (Notebook)

1. **Ingestion & nettoyage**

   * Parsing `Order Date`, extraction `city_state`, `zip`.
   * Agrégations mensuelles par `Product × city_state`.
2. **Feature engineering**

   * Lags (`lag1`, `lag2`), moyennes mobiles (`ma3`, `ma6`), `avg_price`, encodage saisonnier (`sin_month`, `cos_month`).
3. **Modélisation PyCaret**

   * `setup(..., fold_strategy="timeseries")`, `compare_models`, `tune_model`, `finalize_model` par produit.
4. **Prévisions 2025**

   * Boucle récursive mensuelle par ville, initialisation saisonnière si historique court.
   * Sauvegarde CSV “compat” (workaround vieux pandas).
5. **Visualisation géographique**

   * Heatmaps (Folium/Plotly), normalisation min–max, options `log`/`robust`.
6. **Analyse 2025 (prédite)**

   * Tables globales mensuelles, pentes par ville, volatilité, matrices mois×ville.

## Fichiers de sortie (knowledge LLM)

* `predictions_2025_*.csv` : séries mensuelles par ville.
* `timeseries_2025_global.csv` : totals mensuels par produit.
* `geo_trend_volatility_2025.csv` : pente, volatilité, moyenne, somme par ville.
* `city_month_matrix_2025_*.csv` : mois×ville par produit.

## Assistant LLM (Streamlit)

* **But** : répondre aux questions logistiques en s’appuyant *strictement* sur les CSV chargés.
* **Modèle** : *Gemini 2.5 Flash-Lite* via `google-generativeai`.
* **System prompt** : inclut la date du jour et insère dynamiquement les tableaux en Markdown avec résumé (min/max/moyenne).

### Lancer en local

```bash
cd assistant-LLM
# Crée ton venv si besoin, puis :
pip install -r requirements.txt
streamlit run main.py
```

Dans le panneau gauche, renseigne `Clé API Google AI (Gemini)`.

## Docker

### Build

Depuis la racine du projet (contenant `assistant-LLM/` et `outputs/`) :

```bash
docker build -f docker/Dockerfile -t llm-assistant:latest .
```

### Run

```bash
docker run --rm -p 8501:8501 \
  -e GOOGLE_API_KEY="votre_cle" \
  llm-assistant:latest
```

Ouvre [http://localhost:8501](http://localhost:8501).

> Pour monter les `outputs/` dynamiquement :
>
> ```bash
> docker run --rm -p 8501:8501 \
>   -e GOOGLE_API_KEY="votre_cle" \
>   -v "$(pwd)/outputs:/app/outputs:ro" \
>   llm-assistant:latest
> ```
