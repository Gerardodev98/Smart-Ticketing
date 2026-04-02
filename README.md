<div align="center">

# 🎫 Smart Ticketing

**Classificazione e priorità automatica dei ticket aziendali con Machine Learning**

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/sklearn-1.3%2B-orange?logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

*Project Work – Gerardo Di Filippo – Informatica per le Aziende Digitali (L-31)*

</div>

---

## 📌 Panoramica

**Smart Ticketing** è un prototipo minimale e riproducibile che classifica automaticamente i ticket aziendali in tre reparti (Amministrazione, Tecnico, Commerciale) e stima la priorità operativa (alta / media / bassa) tramite tecniche di Machine Learning su testo.

Caratteristiche principali:

- **Dataset sintetico** di 300 ticket generato via script, senza dati personali
- **Due modelli indipendenti** (Multinomial Naive Bayes) per categoria e priorità
- **Preprocessing minimale**: lowercase + rimozione punteggiatura + TF-IDF bigrammi
- **Priorità ibrida**: keyword rules + predizione statistica
- **Dashboard HTML standalone**: si apre nel browser, zero dipendenze aggiuntive
- **Export CSV batch** con predizioni su interi dataset

---

## 📂 Struttura del Repository

```
smart_ticketing/
├── src/
│   ├── genera_dataset.py       # Generatore dataset sintetico (300 ticket)
│   ├── pipeline_ml.py          # Training, valutazione, salvataggio modelli
│   └── genera_dashboard.py     # Genera la dashboard HTML standalone
├── data/
│   └── tickets_sintetici.csv   # Dataset generato (id, title, body, category, priority)
├── models/
│   ├── model_categoria.pkl     # Modello categoria serializzato
│   ├── model_priorita.pkl      # Modello priorità serializzato
│   ├── metrics.json            # Accuracy e F1 dei due modelli
│   └── top_words.json          # Top-5 keyword per categoria
├── assets/
│   ├── confusion_categoria.png # Confusion matrix – categoria
│   ├── confusion_priorità.png  # Confusion matrix – priorità
│   └── f1_bar_chart.png        # Grafico F1 comparativo
├── output/
│   ├── dashboard.html          # Dashboard interattiva (apri nel browser)
│   └── predictions_batch.csv   # Predizioni batch su tutto il dataset
└── README.md
```

---

## 📦 Requisiti

- Python 3.10+
- scikit-learn, pandas, numpy, matplotlib, seaborn

```bash
pip install scikit-learn pandas numpy matplotlib seaborn
```

---

## 🚀 Riproduzione del Progetto

Esegui i tre script nell'ordine seguente:

### Passo 1 – Genera il dataset sintetico

```bash
python src/genera_dataset.py
```

Crea `data/tickets_sintetici.csv` con 300 ticket (100 per categoria).
Le etichette di priorità vengono assegnate automaticamente via keyword.

### Passo 2 – Addestra i modelli e valuta

```bash
python src/pipeline_ml.py
```

- Split 80/20 stratificato per categoria
- Addestra TF-IDF + Naive Bayes per categoria e priorità
- Stampa accuracy, F1-macro e classification report
- Salva i modelli in `models/` e i grafici in `assets/`
- Esporta `output/predictions_batch.csv`

### Passo 3 – Genera la dashboard

```bash
python src/genera_dashboard.py
```

Produce `output/dashboard.html`. **Aprilo direttamente nel browser**, nessun server necessario.

---

## 🖥️ Utilizzo della Dashboard

La dashboard ha quattro sezioni:

| Tab | Contenuto |
|-----|-----------|
| 🔍 Classifica Ticket | Inserisci titolo e descrizione → ottieni categoria, priorità, top-5 keyword e barre di confidenza |
| 📋 Batch & Risultati | Tabella con predizioni sui primi 20 ticket, esportabile in CSV |
| 📊 Metriche del Modello | Accuracy, F1-macro, confusion matrix e bar chart |
| 🗄️ Dataset Sintetico | Anteprima delle prime 10 righe del dataset |

---

## 📊 Risultati

| Modello | Accuracy | F1-macro |
|---------|----------|----------|
| Categoria | 100.0% | 1.000 |
| Priorità | 81.7% | 0.814 |

> Il modello di categoria è perfetto sul dataset sintetico (lessico separato per reparto).
> Il modello di priorità mostra performance più realistiche, con confusione principalmente tra "alta" e "media".

---

## ⚠️ Limiti noti

- Dataset sintetico: le metriche elevate non sono rappresentative di un contesto reale
- Nessuna rimozione stopword italiane
- Nessuna lemmatizzazione
- Priorità basata su keyword fisse, non generalizza a sinonimi

---

## 📄 License

MIT License – vedi [LICENSE](LICENSE)
