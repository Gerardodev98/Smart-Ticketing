"""
genera_dataset.py
-----------------
Genera un dataset sintetico di 300 ticket aziendali.
Colonne output: id, title, body, category, priority
Categorie : Amministrazione | Tecnico | Commerciale
Priorità  : alta | media | bassa  (assegnata via keyword + regole)

Esecuzione:
    python src/genera_dataset.py
Output:
    data/tickets_sintetici.csv
"""

import csv
import random
import os

random.seed(42)

# ── Variabili di riempimento ───────────────────────────────────────────────
MESI   = ["gennaio","febbraio","marzo","aprile","maggio","giugno",
          "luglio","agosto","settembre","ottobre","novembre","dicembre"]
NUMERI = [f"{n:04d}" for n in range(1000, 1200)]
IMPORTI= [120, 250, 480, 750, 1200, 2300, 340, 89, 670, 990]

def r(lst):
    return random.choice(lst)

def fill(s):
    return (s.replace("{mese}", r(MESI))
             .replace("{num}",  r(NUMERI))
             .replace("{importo}",  str(r(IMPORTI)))
             .replace("{importo2}", str(r(IMPORTI)))
             .replace("{giorno}",   f"{random.randint(1,28)}/{random.randint(1,12)}/2024"))

# ── Template lessicali per categoria ──────────────────────────────────────
TEMPLATES = {

    "Amministrazione": {
        "titles": [
            "Richiesta fattura {mese}",
            "Mancato rimborso spese {mese}",
            "Errore importo su fattura n. {num}",
            "Chiarimento su nota credito",
            "Scadenza pagamento non ricevuta",
            "Fattura duplicata {mese}",
            "Aggiornamento dati fiscali azienda",
            "Sollecito pagamento arretrato",
            "Richiesta ricevuta di pagamento",
            "Discrepanza importo in estratto conto",
        ],
        "bodies": [
            "Buongiorno, non ho ancora ricevuto la fattura relativa al mese di {mese}. Potete inviarla al più presto?",
            "La fattura numero {num} riporta un importo errato di {importo} euro invece di {importo2}. Chiedo rettifica.",
            "Ho effettuato il pagamento il {giorno} ma non risulta registrato. Allego la contabile bancaria.",
            "Sono in attesa del rimborso spese presentato a {mese}. Trascorsi 30 giorni, sollecito l'evasione.",
            "Riscontro una nota credito non applicata sull'ultima fattura. Chiedo verifica e storno corretto.",
            "Non ho ricevuto il promemoria di scadenza per la rata di {mese}. Inviatemi le istruzioni di pagamento.",
            "La fattura {num} risulta emessa due volte nel portale. Chiedo la cancellazione del duplicato.",
            "I dati fiscali sono cambiati. Allego il nuovo modello di registrazione per l'aggiornamento.",
            "Solicito ricevuta di pagamento relativa alla transazione del {giorno} per {importo} euro.",
            "L'estratto conto mensile non corrisponde alle fatture. Chiedo un controllo dettagliato.",
        ],
    },

    "Tecnico": {
        "titles": [
            "Sistema bloccato dopo aggiornamento",
            "Errore critico all'avvio dell'applicazione",
            "Stampante di rete non raggiungibile",
            "VPN aziendale non funzionante",
            "Perdita dati su drive condiviso",
            "PC lento e instabile",
            "Schermata blu ripetuta sul laptop",
            "Impossibile accedere al portale interno",
            "Malfunzionamento scanner documenti",
            "Email non inviate: errore SMTP",
        ],
        "bodies": [
            "Dopo l'aggiornamento di ieri il software è bloccato. Non riesco ad aprire nessuna funzione.",
            "All'avvio ricevo 'Errore critico: database non raggiungibile'. Il servizio è fermo da stamattina.",
            "La stampante condivisa non risponde. Altri colleghi hanno lo stesso problema.",
            "La VPN non si connette: errore timeout. Ho già riavviato il computer ma il problema persiste.",
            "Alcuni file sul drive condiviso sono scomparsi. È urgente il ripristino dei dati.",
            "Il PC è lento nell'ultima settimana: avvio lungo, applicazioni bloccate, disco al 100%.",
            "Il laptop va in schermata blu ogni ora. Temo ci sia un driver corrotto o un problema hardware.",
            "Il portale restituisce errore 403. Ho provato con più browser ma la situazione non cambia.",
            "Lo scanner non viene riconosciuto dopo la sostituzione del cavo USB. Ho reinstallato i driver.",
            "Le email non vengono inviate: errore 'Connessione SMTP rifiutata'. Riguarda tutti gli utenti.",
        ],
    },

    "Commerciale": {
        "titles": [
            "Richiesta preventivo licenze software",
            "Stato dell'ordine n. {num}",
            "Offerta scaduta: rinnovo contratto",
            "Promozione non applicata all'ordine",
            "Richiesta catalogo prodotti aggiornato",
            "Domanda su condizioni di fornitura",
            "Consegna in ritardo ordine {num}",
            "Interesse per piano Enterprise",
            "Modifica quantità su ordine in corso",
            "Richiesta referenze commerciali",
        ],
        "bodies": [
            "Siamo interessati ad acquistare 50 licenze. Potete inviarci un preventivo dettagliato?",
            "L'ordine {num} risulta ancora in lavorazione. A quando è prevista la spedizione?",
            "Il contratto scade a fine mese. Vorrei ricevere la proposta di rinnovo con le nuove condizioni.",
            "La promozione del 15% non è stata applicata all'ordine {num}. Chiedo correzione.",
            "Vorrei ricevere il catalogo aggiornato con i nuovi prodotti del secondo semestre.",
            "Quali sono le condizioni minime di fornitura per un ordine ricorrente mensile?",
            "L'ordine {num} doveva arrivare il {giorno} ma non è ancora stato consegnato. Verificate.",
            "Vorrei capire i vantaggi del piano Enterprise rispetto all'abbonamento Standard attuale.",
            "Devo modificare la quantità dell'ordine {num} da 10 a 25 unità. È ancora possibile?",
            "Potete fornirmi alcune referenze commerciali di aziende che usano il vostro servizio?",
        ],
    },
}

# ── Regole keyword per priorità ───────────────────────────────────────────
PRIORITY_RULES = {
    "alta": [
        "bloccato", "critico", "urgente", "fermo", "perdita", "schermata blu",
        "non funzionante", "errore critico", "ripristino", "scomparsi",
        "bloccata", "blocco", "emergenza", "virus", "corrotto",
    ],
    "media": [
        "ritardo", "mancato", "non ricevuto", "sollecito", "discrepanza",
        "duplicata", "non applicata", "lento", "instabile", "non raggiungibile",
        "non riconosciuto", "non risulta", "ancora in lavorazione",
    ],
}

def assegna_priorita(title: str, body: str) -> str:
    testo = (title + " " + body).lower()
    for kw in PRIORITY_RULES["alta"]:
        if kw in testo:
            return "alta"
    for kw in PRIORITY_RULES["media"]:
        if kw in testo:
            return "media"
    return "bassa"

# ── Generazione ────────────────────────────────────────────────────────────
def genera_ticket(n_per_category: int = 100) -> list[dict]:
    tickets = []
    ticket_id = 1
    for cat, tpl in TEMPLATES.items():
        for _ in range(n_per_category):
            title = fill(r(tpl["titles"]))
            body  = fill(r(tpl["bodies"]))
            priority = assegna_priorita(title, body)
            tickets.append({
                "id":       ticket_id,
                "title":    title,
                "body":     body,
                "category": cat,
                "priority": priority,
            })
            ticket_id += 1
    random.shuffle(tickets)
    # ri-assegna id ordinato dopo lo shuffle
    for i, t in enumerate(tickets, 1):
        t["id"] = i
    return tickets

def main():
    os.makedirs("data", exist_ok=True)
    tickets = genera_ticket(100)   # 300 totali
    out_path = "data/tickets_sintetici.csv"
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["id","title","body","category","priority"])
        writer.writeheader()
        writer.writerows(tickets)
    print(f"[OK] Dataset generato: {out_path} ({len(tickets)} righe)")
    # stampa distribuzione
    from collections import Counter
    cats = Counter(t["category"] for t in tickets)
    pris = Counter(t["priority"] for t in tickets)
    print(f"     Categorie : {dict(cats)}")
    print(f"     Priorità  : {dict(pris)}")

if __name__ == "__main__":
    main()
