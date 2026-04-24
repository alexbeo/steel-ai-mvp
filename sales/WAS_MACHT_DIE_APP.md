# Was macht diese Anwendung — in einfachen Worten

Kurze und ehrliche Beschreibung für Nicht-Metallurgen: was das Programm tut, wie, und welchen Nutzen es bringt. Kein Marketing, keine Erfindungen — nur das, was tatsächlich im Code vorhanden ist.

---

## In zwei Sätzen

Das ist ein Programm, das einem **Metallurgen** dabei hilft, die **chemische Zusammensetzung eines Stahls** so zu wählen, dass er die geforderte Festigkeit erreicht und gleichzeitig so günstig wie möglich wird. Zusätzlich enthält es einen physikalischen Rechner für eine konkrete Operation im Pfannenofen — die **Desoxidation mit Aluminium**.

---

## Etwas Kontext (falls Sie kein Metallurge sind)

Stahl ist eine Legierung aus Eisen mit Kohlenstoff und weiteren Zusätzen (Mangan, Nickel, Niob, Vanadium, Titan…). Die Eigenschaften eines Stahls — Festigkeit, Härte, Duktilität, Schweißbarkeit — hängen davon ab, **wieviel und was** hinzugefügt wurde und **wie der Stahl danach gewalzt und abgekühlt** wurde.

Die Aufgabe eines Werksingenieurs: Ein Kunde verlangt zum Beispiel „ein Rohr mit einer Streckgrenze zwischen 485 und 580 MPa" (Sorte X65 für Pipelines). Der Ingenieur muss ein Rezept entwickeln: wieviel Kohlenstoff, Mangan, Niob usw. in die Schmelze kommen.

Heute geschieht das im Wesentlichen **durch Probieren**: eine Versuchsschmelze herstellen, prüfen, bei Nichterfüllung die nächste. Eine Schmelze = Tage Arbeit und erhebliche Kosten. Hinzu kommt, dass die Preise von Ferrolegierungen (insbesondere Niob aus Brasilien) stark schwanken — ein gestern „gutes" Rezept kann heute anderthalbmal so teuer sein.

---

## Was die Anwendung tut — in fünf Reitern

Die Anwendung läuft im Browser und hat 5 Reiter:

### 1. „Modelltraining"

Das Programm nimmt einen **Datensatz** (eine Tabelle: „diese Zusammensetzung + diese Wärmebehandlung → ergab diese Eigenschaft") und trainiert darauf ein mathematisches Modell. Das Modell ist der Algorithmus `XGBoost` (ein in Machine Learning sehr verbreitetes Verfahren). Nach dem Training kann es die Eigenschaft aus der Zusammensetzung vorhersagen.

**Wichtige Einschränkung:** Der Datensatz ist aktuell **synthetisch** — das Programm erzeugt ihn selbst anhand physikalischer Formeln. Echte Schmelzdaten aus einem realen Werk sind noch nicht eingespielt. Für einen produktiven Einsatz muss man den Datensatz durch die Daten eines konkreten Werks ersetzen.

### 2. „Prognose"

Der Ingenieur gibt eine Zusammensetzung ein (C = 0,08 %, Mn = 1,5 %, Nb = 0,025 %…) sowie die Walzparameter. Das Programm liefert:

- den vorhergesagten Wert der Eigenschaft (z. B. σ_t = 548 MPa)
- ein **Konfidenzintervall** (z. B. ±18 MPa). Das ist ehrlich: Das Programm gibt nicht eine einzige Zahl, sondern sagt „mit 90 % Wahrscheinlichkeit liegt der Wert im Bereich 530–566".
- eine Warnung „diese Zusammensetzung liegt außerhalb dessen, worauf das Modell trainiert wurde — der Prognose nicht blind vertrauen" (Out-of-Distribution-Flag).

### 3. „Legierungsdesign" — das eigentliche Wertangebot

Der Ingenieur macht die Aufgabe **umgekehrt**: Er sagt dem Programm „ich brauche einen Stahl mit σ_t zwischen 485 und 580 MPa, Schweißbarkeit CEV ≤ 0,43". Das Programm startet einen **Optimierungsalgorithmus** (`NSGA-II` — aus der evolutionären Biologie entlehnt, gleiches Prinzip wie Selektion und Kreuzung). In 10–60 Sekunden werden tausende Zusammensetzungen durchprobiert, und die **Top 5** werden ausgegeben — sortiert nach **Einsatzkosten in EUR pro Tonne**.

Die Einsatzkosten werden anhand realer Ferrolegierungspreise berechnet, die in das Programm geladen sind (FeMn, FeSi, FeNb, FeV, FeTi usw. — insgesamt 11 Positionen). Die Preise lassen sich direkt in einer Tabelle anpassen, und die Rechnung wird aktualisiert.

**Dieser Teil funktioniert zurzeit nur für HSLA-Rohrstähle (API 5L).** Für Q&T-Kohlenstoffstähle noch nicht.

### 4. „Desoxidation"

Ein separater Rechner für eine konkrete Operation im Pfannenofen. Nach dem Schmelzprozess ist im Stahl viel gelöster **Sauerstoff** — dieser muss „abgebunden" werden, indem Aluminium zugegeben wird. Zu wenig Al → der Stahl wird spröde. Zu viel Al → Verschwendung eines teuren Zusatzes plus unerwünschte Oxideinschlüsse.

Das Programm berechnet aus dem Sauerstoffmesswert (ppm), der Temperatur und der Metallmasse, **wieviel Kilogramm Aluminium** zugegeben werden müssen. Es verwendet dafür **drei unterschiedliche wissenschaftliche Modelle** (Fruehan 1985, Sigworth-Elliott 1974, Hayashi-Yamamoto 2013) und zeigt alle drei Ergebnisse. Sind sie kongruent, kann der Empfehlung vertraut werden. Weichen sie ab, so ist die zugrundeliegende Physik selbst unpräzise — dann muss der Ingenieur selbst entscheiden.

### 5. „Historie"

Jede im Programm getroffene Entscheidung (Modelltraining, Designlauf, Desoxidationsberechnung) wird **automatisch in ein Protokoll geschrieben** (eine SQLite-Datei). In einem Jahr lässt sich nachvollziehen: „Warum haben wir im März dieses Rezept gewählt? Wie waren die Preise? Welche Alternativen wurden verworfen?"

---

## Was sonst noch „unter der Motorhaube" steckt (kurz)

- **Pattern Library** — eine Checkliste von 22 typischen Fehlern, aufgeteilt nach Pipeline-Phasen: 5 Daten-Checks (Target Leakage, Distribution Shift, uneinheitliche Einheiten, Random Split in Zeitreihen, Überschreitung der physikalischen Zusammensetzungsgrenzen), 6 Modell-Checks (Overfitting, schlechte CI-Kalibrierung, fehlende Unsicherheit, unplausible Feature-Wichtigkeit, fehlender OOD-Detektor, falsche Cross-Validation-Schema), 3 Checks für inverse Rezeptursuche, 4 für Kostenmodellierung, 3 für Al-Desoxidation, 1 für Validierung. Nach jeder Phase (Training / Design / Desoxidation / Validierung) läuft der aktuelle Zustand gegen die relevanten Patterns und bekommt entweder „ok" oder eine Warnung; im strengen Modus stoppt die Pipeline bis zur Benutzerantwort.
- **LLM-Critic** — optional: Wenn ein Anthropic-API-Schlüssel hinterlegt ist, liest Claude Sonnet nach dem Training das Ergebnis und verfasst Beobachtungen. Das ist eine zweite Meinung durch eine KI.

---

## Welcher konkrete Nutzen entsteht

Ich werde nicht mit „Millionen gesparter Euro" fantasieren — das wären Schätzungen, keine Messungen. Die ehrlichen Vorteile:

1. **Geschwindigkeit.** Ein Rezept entsteht in 30–90 Sekunden statt in Tagen manueller Excel- und Handbuchrechnung.
2. **Ehrliche Unsicherheit.** Das Programm sagt nicht „es werden genau 550 MPa", sondern liefert einen Bereich, in dem die Eigenschaft mit 90 % Wahrscheinlichkeit liegt. Das senkt die Quote an Schmelzen, die die Spezifikation verfehlen.
3. **Berücksichtigung der Preise.** Der Metallurge denkt gewöhnlich über Eigenschaften nach und vergisst die Kosten; der Einkauf denkt über Preise nach und versteht die Metallurgie nicht. Das Programm hält beide Achsen gleichzeitig.
4. **Gedächtnis.** In einem halben Jahr, wenn gefragt wird „warum habt ihr damals dieses Rezept gewählt", muss nicht im Postfach gesucht werden — man öffnet das Protokoll und sieht alles: Preise, Constraints, Alternativen.
5. **Bremsen gegen typische ML-Fehler.** Die integrierte Checkliste verhindert, dass das Modell „schummelt" — z. B. unrealistische Werte vorhersagt oder weit über den Trainingsbereich hinaus extrapoliert.

---

## Was es **nicht** tut (wichtig, nicht zu übertreiben)

- Es ersetzt keinen erfahrenen Metallurgen. Es ist ein **Assistent**, kein Automat.
- Es funktioniert nicht „aus der Box" auf realen Werksdaten — der Kundendatensatz muss eingespielt werden.
- Es berechnet keinen Carbon Footprint, ist nicht CBAM-ready und verfolgt nicht den eingebetteten CO₂ der Ferrolegierungen. Das sind potenzielle zukünftige Funktionen, im Code aktuell nicht enthalten.
- Es ist nicht mit Enterprise-Systemen (ERP / MES) integriert. Es handelt sich um eine lokale Webanwendung für einen einzelnen Anwender.
- Das inverse Design arbeitet zurzeit nur für eine Stahlklasse (HSLA-Rohrstähle). Für Q&T-Kohlenstoffstähle sind nur Prognose und Training verfügbar — kein Rezeptvorschlag.
