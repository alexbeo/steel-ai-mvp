# Steel AI — Pitch Deck
## voestalpine AG · Einkaufsleitung Ferrolegierungen & Rohstoffe

**Vertraulich · April 2026**

Adressat: Einkauf Konzernebene · voestalpine AG, Linz
Sprache: Deutsch · Format: 14 Folien · Lesezeit: 8–10 Minuten

---

## Folie 1 — Titel

# Steel AI

## Physik-informierter KI-Copilot für den Legierungseinkauf

**EUR/t, nicht MPa — KI, die Ihre Beschaffungsstrategie versteht.**

---

Vorgelegt: [Monat] 2026
Kontakt: [Gründername] · [E-Mail] · [Telefon]
Steel AI · Liefer- und Rechnungsadresse EU

---

## Folie 2 — Drei Kräfte, die den Stahleinkauf 2026 neu definieren

**1. Rohstoffvolatilität ohne Präzedenz**
FeMo +180% zwischen 2021 und 2024. FeNb — CBMM (Brasilien) kontrolliert rund 75% der globalen Primärproduktion. FeV schwankt regelmäßig ±40%. Excel-Rezepturen aus den 1990er Jahren reagieren zu langsam auf Spot-Preis-Bewegungen.

**2. Generationenwechsel im metallurgischen Know-how**
Eine Generation erfahrener Legierungs-Entscheider geht in den kommenden fünf bis zehn Jahren in den Ruhestand. Tacit knowledge — Bauchgefühl, Anekdoten, „so machen wir das" — verlässt die Organisation schneller, als es formalisiert wird. Ohne kodifiziertes Entscheidungssystem wird aus jedem Personalwechsel ein Wissensverlust.

**3. greentec steel, H₂-DRI, EAF-lastige Chargen**
Neue Einsatzmaterialien verlangen neue Legierungsrezepte. Handbuch-Formeln (CEV 1967, Pcm 1968) bleiben valide — wurden aber nie für Spot-Preis-Optimierung auf geänderten Charge-Profilen konzipiert. Linz und Donawitz gehen 2027 in Betrieb.

> **Konsequenz:** Einkaufsentscheidungen müssen vier Dimensionen gleichzeitig halten — Zusammensetzung × Kosten × Volatilitätsrisiko × Spezifikationserfüllung — und dabei das Know-how der Organisation über Personalwechsel hinweg bewahren. Heute geschieht das sequenziell, in getrennten Tools, mit einem Lag von Tagen.

---

## Folie 3 — Wettbewerbslücke

| System | Zweck | Was für den Einkauf fehlt |
|---|---|---|
| Excel + Handbuch | Rezepturberechnung (CEV, Pcm) | Kein Live-Preisbezug, keine Volatilitäts-Abwägung, kein Audit Trail |
| ERP / Procurement-Systeme | Lieferantensteuerung, Hedging | Kennt Metallurgie nicht |
| MES / Level-2 | Prozesssteuerung | Kennt Kosten nicht |
| Citrine Informatics (US, Series C) | Materialentwicklung für Chemie/CPG | Nicht vertikal Stahl; nicht einkaufsorientiert |
| QuesTek ICMD (US) | Legierungsdesign Aerospace/AM | HSLA und Baustähle nicht Fokus |
| Intellegens Alchemite (UK) | Generisches ML-Tool | Benötigt Data-Science-Kompetenz beim Kunden |

**White space:** eine einkaufsseitige Entscheidungsschicht, die Metallurgie, Preise, Volatilitäts-Szenarien und Auditierbarkeit in einem Workflow vereint.

Dort ist Steel AI positioniert — und nur dort.

---

## Folie 4 — Unsere These

> *„Die beste Zusammensetzung ist nicht die metallurgisch eleganteste —
> sondern jene, die Ihre Spezifikation bei niedrigsten Total-Landed-Costs erfüllt,
> unter realer Rohstoffpreislage und unter Berücksichtigung des Volatilitätsrisikos."*

Diese These haben wir operationalisiert — als Software, die eine Einkäuferin ohne Data-Science-Hintergrund in drei Minuten bedient.

---

## Folie 5 — Plattform auf einen Blick

```
┌────────────────────────────────────────────────────────────┐
│  INPUT   Zielspezifikation    Rohstoffpreis-Snapshot       │
│          (z.B. API 5L X65)    (EUR/kg, live oder YAML)     │
│                               11 Ferrolegierungen + Schrott│
└────────────────────────────────────────────────────────────┘
                             ↓
┌────────────────────────────────────────────────────────────┐
│  ENGINE                                                    │
│  • Forward-ML         XGBoost + Quantile Regression        │
│                       (Eigenschaften mit 90%-CI)           │
│  • Inverse Design     NSGA-II Multi-Objective              │
│                       (Pareto: Eigenschaft × Kosten ×      │
│                        Schweißbarkeit)                     │
│  • Physik-Validator   CEV, Pcm, CEN, Schweißbarkeit,       │
│                       Heißverformung, OOD-Detection        │
│  • Critic-Schicht     Pattern Library (20+ Anti-Patterns)  │
│                       + Claude Sonnet LLM-Review           │
└────────────────────────────────────────────────────────────┘
                             ↓
┌────────────────────────────────────────────────────────────┐
│  OUTPUT  Top-5 Pareto-Kandidaten                           │
│          je mit: Chemie, Prozess, σ_t ± CI,                │
│          EUR/t-Breakdown je Ferrolegierung,                │
│          OOD-Flag, nächste EN/API-Güte                     │
└────────────────────────────────────────────────────────────┘
                             ↓
┌────────────────────────────────────────────────────────────┐
│  AUDIT   Decision Log (SQLite, EU-resident)                │
│          Jede Anfrage mit Snapshot: Preise, Constraints,   │
│          Begründung, Alternativen, Autor. Nach Jahren      │
│          rekonstruierbar. Compliance-ready.                │
└────────────────────────────────────────────────────────────┘
```

Laufzeit Ende-zu-Ende: 30–90 Sekunden auf Standard-Hardware. Kein GPU-Cluster nötig.

---

## Folie 6 — Live-Demo: Preisschock-Szenario (FeNb)

**Ausgangssituation:** FeNb-Preis steigt von 36 auf 54 EUR/kg (+50%) — etwa infolge einer CBMM-Supply-Disruption oder eines brasilianischen Exportzolls.

**Vor dem Schock — Rezeptur A (Nb-dominiert)**

| Parameter | Wert |
|---|---|
| Chemie | C 0,080% · Mn 1,45% · **Nb 0,025%** · V 0,003% |
| σ_t Prognose | 548 ± 18 MPa *(Ziel API 5L X65: 485–580)* |
| CEV(IIW) / Pcm | 0,34 / 0,18 |
| **Einsatzkosten** | **428 EUR/t** |

**Nach dem Schock — Rezeptur B, neu berechnet in 45 Sekunden**

| Parameter | Wert |
|---|---|
| Chemie | C 0,085% · Mn 1,55% · Nb 0,015% · **V 0,035% · Ti 0,025%** |
| σ_t Prognose | 551 ± 21 MPa *(Spezifikation weiter erfüllt)* |
| CEV(IIW) / Pcm | 0,35 / 0,18 |
| **Einsatzkosten** | **419 EUR/t** |

**Delta: −9 EUR/t ohne Spezifikationsverlust.**

Hochgerechnet auf voestalpine Tubulars GmbH (Kindberg, nahtlose Rohre API 5L) bei einer Jahresleistung im sechsstelligen Tonnenbereich entspricht das bereits bei 500.000 t/Jahr einer **Entlastung von 4,5 Mio. EUR pro Volatilitätsepisode**. FeNb-Volatilität tritt historisch mehrmals pro Jahr auf.

Jede Optimierung: rund 30 Sekunden, auf dem Laptop einer Einkäuferin, ohne F&E-Ticket.

---

## Folie 7 — Wissenschaftliche Grundlage

Keine Black-Box-Magie. Jede Funktion basiert auf peer-reviewed Methoden. Ein voestalpine-Technologe kann jede Formel in 30 Minuten nachrechnen.

**Thermodynamik — Al-Desoxidation im Pfannenofen**
- Fruehan, R.J. (Carnegie Mellon University) — *The Making, Shaping and Treating of Steel*, 11. Auflage, AISE Steel Foundation, 1998. Standardreferenz der US-Stahlindustrie.
- Sigworth, G.K.; Elliott, J.F. (MIT) — *Metal Science* 8 (1974) — Interaction-Parameter-Formalismus.
- Hayashi, M.; Yamamoto, H. — *ISIJ International* 53 (2013) — Hochpräzisionsmodell für [Al] > 0,05%.

*Alle drei Modelle sind parallel implementiert. Der Anwender sieht die Streuung als ehrliche Unsicherheitsangabe, nicht als Single-Number-Pseudogenauigkeit.*

**Schweißbarkeit — Industry Standard**
- CEV nach IIW-Doc. IX-535-67 (International Institute of Welding, 1967)
- Pcm nach Ito, Y.; Bessyo, K. — *Journal of the Japan Welding Society* 37 (1968)
- CEN nach Yurioka, N. — *Welding in the World* 28 (1990)

**Machine Learning**
- XGBoost — Chen, T.; Guestrin, C. — *KDD 2016*. Über 45.000 Zitationen.
- NSGA-II — Deb, K. et al. — *IEEE Transactions on Evolutionary Computation* 6 (2002). Über 50.000 Zitationen. De-facto-Standard für Multi-Objective-Optimierung.
- SHAP — Lundberg, S.; Lee, S.-I. — *NeurIPS 2017*. Erklärbarkeit einzelner Vorhersagen auf Feature-Ebene.
- Conformal Prediction — Vovk, V. et al., Springer 2005. Kalibrierte 90%-Konfidenzintervalle mit mathematischer Abdeckungsgarantie.

> **Konsequenz:** Der Wert entsteht aus der **Integration** dieser Methoden in einen einkaufsfähigen Workflow — nicht aus versteckter IP. Transparenz ist hier ein Feature, kein Zugeständnis.

---

## Folie 8 — Vier Wertvektoren für Ihren Einkauf

**1. Hebel gegen das CBMM/Niob-Monopol**
CBMM kontrolliert rund 75% der globalen Nb-Primärproduktion. Jeder Preisausschlag bei FeNb trifft hochfeste Rohr-Güten unmittelbar. Steel AI dimensioniert den Substitutionskorridor Nb → V/Ti in Echtzeit — einkaufsseitig, nicht als F&E-Projekt mit Sechs-Monats-Laufzeit. **Sofort anwendbar in voestalpine Tubulars (API 5L X60–X80).**

**2. Al-Desoxidations-Copilot für den Pfannenofen**
Zweiter unabhängiger Hebel in derselben Plattform: Forward- und Inverse-Rechner für die Al-Zugabe im Ladle Furnace auf Basis **dreier parallel implementierter thermodynamischer Modelle** (Fruehan 1985, Sigworth-Elliott 1974, Hayashi-Yamamoto 2013). Die Streuung zwischen den Modellen ist die ehrliche Unsicherheitsangabe. Der Einkauf kann die Al-Zugabe gegen die **effektive Al-Ausbeute einer Lieferung** gegenrechnen — Qualitätsbewertung der Al-Lieferanten wird quantitativ, nicht anekdotisch.

**3. Lieferantendiversifizierung gegen Klumpenrisiken**
Szenariosimulation auf Knopfdruck: *„Was, wenn wir 40% des FeMo-Volumens von Lieferant X auf Lieferant Y umstellen — bei +8% Preis, aber Lead-Time 12 → 4 Wochen?"* — Antwort in 2 Minuten mit voller Wirkungsschätzung auf alle aktiven Legierungsrezepte. Nicht drei Wochen Excel.

**4. Prüfungssicherer Audit-Trail**
Jede ausgelöste Optimierung wird mit Preis-Snapshot, Constraints, Top-5-Alternativen, Begründung und Autor in einer SQLite-Datenbank festgeschrieben. Nach zwei Jahren im internen oder externen Audit rekonstruierbar: *„Warum wurde im März 2027 die teurere Nb-haltige Variante gewählt?"* — Antwort in 30 Sekunden, nicht in drei Wochen Aktenarchäologie.

---

## Folie 9 — Pilot-Vorschlag: 90-Tage-Proof

Zwei Stufen mit Null-Risiko-Einstieg.

**Stufe 0 — Benchmark-Audit (14 Tage · kostenfrei)**

1. Sie übergeben uns unter NDA drei bis fünf anonymisierte aktuelle Legierungsrezepturen (empfohlen: aus voestalpine Tubulars, API 5L X60/X65/X70).
2. Wir rechnen sie mit Ihren heutigen Ferrolegierungspreisen (oder unserem geprüften Seed-Benchmark) durch die Plattform.
3. Sie erhalten einen Report: konkretes EUR/t-Einsparpotenzial je Güte, Volatilitäts-Sensitivitäten, Substitutionskorridor-Analyse.
4. Entscheidung: Pilot ja/nein — mit realen Zahlen zu Ihren eigenen Rezepturen, nicht mit Versprechen.

**Stufe 1 — Pilot (90 Tage · 85.000 EUR all-in)**

- Preisdaten via CSV- oder YAML-Snapshot (keine ERP-Integration erforderlich)
- Kundenspezifisches Modell auf einer Stahlklasse (Empfehlung: API 5L X60–X70 Tubulars)
- 10 reale Anfragen Ihres Einkaufs-/F&E-Teams über die Plattform
- Wöchentliche 30-min Lenkungstermine, finaler ROI-Bericht auf Vorstandsebene
- Training: zwei Tage für Ihr Team vor Ort in Linz

**Erfolgskriterien (zwei von drei müssen erfüllt sein):**
1. ≥ 5 EUR/t quantifizierte Einsparung auf der Pilot-Stahlklasse bei gleicher oder besserer Spezifikationserfüllung
2. ≥ 3× schnellere Entscheidungszeit gegenüber dem heutigen Excel-Workflow (gemessen: Anfrage → Empfehlung)
3. Resilienz-Bewertung bei ≥ 3 Volatilitätsszenarien (FeNb, FeMo, Al-Lieferanten) als steuerungsfähig akzeptiert

---

## Folie 10 — Kommerzielle Optionen

**Option A — Fixpreis-Pilot: 85.000 EUR netto**
Klar kalkulierbar. Zahlungsplan: 40% bei Kickoff · 30% nach Phase 2 · 30% bei Abschluss.
*Geeignet bei hoher Budgetsicherheits-Anforderung.*

**Option B — Erfolgsbasiert: 40.000 EUR Basis + 60.000 EUR Erfolgsbonus**
Basis zahlbar monatlich über 6 Monate. Bonus nur bei Erfüllung ≥ 2 der 3 Erfolgskriterien.
*Geeignet bei Anreiz-Fokus — Steel AI hat Skin in the Game.*

**Option C — Strategische Partnerschaft: 50.000 EUR + Mitentwicklungsrechte**
Gemeinsam entwickelte Güte (z.B. H₂-ready X80 für greentec steel) bleibt voestalpine-IP. Steel AI erhält 24 Monate Segment-Exklusivität und Referenzrecht.
*Geeignet bei strategischer Tiefe und minimaler Cash-Ausgabe.*

**Add-ons (optional)**
- On-Premise-Deployment im voestalpine-Rechenzentrum: +30.000 EUR einmalig
- Jede zusätzliche Stahlklasse: +25.000 EUR
- Folgelizenz nach Pilot (bei positivem Ergebnis): 120.000–200.000 EUR p.a., je nach Scope

---

## Folie 11 — Datenschutz, IP und Betriebskontinuität

**Daten**
- Ihre Plaven-/Prozessdaten verbleiben in Ihrem Tenant. Hosting nach Wahl: AWS eu-central-1 (Frankfurt) oder on-premise im voestalpine-Rechenzentrum.
- Training ausschließlich Ihrer Modellgewichte, keine Querverwendung mit anderen Kunden.
- Verschlüsselung at-rest (AES-256) und in-transit (TLS 1.3). DSGVO-konform nach Art. 28 (Auftragsverarbeiter). AVV-Entwurf auf Anfrage.

**Intellectual Property**
- Durch die Plattform entwickelte Zusammensetzungen = 100% voestalpine-Eigentum. Keine Lizenzen, keine Royalties.
- Steel AI-Methodologie (Agenten-Architektur, Pattern Library, Decision Log) = Steel AI-Eigentum.
- Gemeinsame Publikationen und Fallstudien nur mit schriftlicher Freigabe.

**Sicherheit & Compliance**
- CIS Controls v8 Level 2 heute implementiert. ISO 27001 Zertifizierung geplant Q4 2026.
- Pen-Test-Bericht eines akkreditierten DACH-Anbieters auf Anfrage.
- NIS2-konforme Meldungspfade.

**Exit-Schutz**
- Alle Modellartefakte exportierbar als ONNX (offener Standard).
- Decision Log als Standard-SQLite-Datei übergebbar.
- 30-Tage-Kündigungsklausel ohne Begründung. Keine Vertragsstrafen.

---

## Folie 12 — Warum Steel AI, warum jetzt

**Team**

- **[Gründername]** · Gründer & CEO. Zuvor Metal Trading (praktisches Verständnis des Rohstoffgeschäfts), zuletzt drei Jahre in angewandter KI/LLM-Entwicklung.
- **[Deutscher Co-Founder]** · Domain Lead. [N] Jahre europäische Stahl-F&E. Netzwerk: voestalpine Linz/Donawitz, Salzgitter, SMS group, Primetals.
- **Beirat (im Ausbau):** ordentlicher Professor Metallurgie einer DACH-Technischen Universität; ehemaliger CTO eines europäischen Stahlwerks.

**Warum jetzt**

- **Operativ — Volatilität ist akut spürbar**: FeMo +180% zwischen 2021 und 2024, regelmäßige ±40%-Ausschläge bei FeV und FeNb. Ad-hoc-Lösungen in Excel stoßen sichtbar an Grenzen. Der Leidensdruck existiert heute, nicht erst in drei Jahren.
- **Foundation Models ausgereift**: Claude Sonnet 4.6 und Opus 4.7 ermöglichen 2026 erstmals produktionsreife Agenten-Orchestrierung. Was 2023 Forschung war, ist heute Infrastruktur.
- **Demografisch — Know-how-Transfer**: Die tragende Generation der Legierungs-Entscheider geht in den kommenden fünf bis zehn Jahren in den Ruhestand. Wer jetzt ein Entscheidungssystem aufbaut, transferiert das Wissen; wer wartet, verliert es.

**Kapazitätsgrenze**
Wir nehmen im Geschäftsjahr 2026 **maximal zwei Pilot-Partner** an, um die volle Aufmerksamkeit der Gründer auf jedes Projekt zu sichern. Ein Slot ist bereits in aktiven Gesprächen.

---

## Folie 13 — Konkreter nächster Schritt

Fünf-Stufen-Pfad, jeder Schritt niederschwellig.

| Schritt | Wann | Aufwand Ihrerseits |
|---|---|---|
| 1. Kurzes Feedback (Interesse ja/nein) | Diese Woche | 5 Min · eine E-Mail |
| 2. 30-min exploratorisches Gespräch (MS Teams) | Woche 2 | 30 Min · kein NDA erforderlich |
| 3. NDA + Sample-Daten (3–5 anonymisierte Rezepturen) | Woche 3 | 2 Std Einkauf/F&E |
| 4. Kostenloser Benchmark-Audit mit EUR/t-Potenzial | Woche 5 | 0 — wir liefern |
| 5. Entscheidung Pilot ja/nein | Woche 6 | Interner Entscheidungsprozess |

**Kontakt**

[Gründername] · Steel AI
E-Mail: [E-Mail einfügen]
Tel: [Telefon einfügen]
Liefer- und Rechnungsadresse EU

---

**PS.** Wenn Stufe 0 — der kostenlose zweiwöchige Benchmark-Audit — keinen substanziellen Mehrwert für Ihre konkreten Rezepturen zeigt, endet die Beziehung ohne Folgekosten. Das ist die einzige ehrliche Art, einen Beweis zu führen.

---

## Folie 14 — Anhang: Technische Vertiefung

Für das technische Gespräch mit Ihrem Metallurgie-/IT-Team auf Abruf bereit:

- **A.1** Modellarchitektur im Detail (Forward + Inverse + Validator)
- **A.2** Pattern Library — 20+ ML-Anti-Patterns mit Testabdeckung (Data Leakage, Calibration Drift, OOD, physikalische Konsistenz, Cost-Model-Integrität)
- **A.3** Beispielexport Decision Log (JSON/SQLite-Schema)
- **A.4** Al-Desoxidations-Modul: Forward/Inverse/Modellvergleich (LF-Advisory für Stahlwerk Linz/Donawitz)
- **A.5** Wettbewerbsmatrix (Citrine, QuesTek, Intellegens, MatMinds) mit Evaluation Criteria
- **A.6** Preisdaten-Workflow (YAML-Snapshot-Format · CSV-Upload · Versionierung im Decision Log)
- **A.7** Pen-Test-Summary und SecOps-Runbook
- **A.8** LLM-Critic-Dokumentation (Claude Sonnet 4.6 Integration, Prompt Caching, Token-Ökonomie)

---

*Ende des Decks · 14 Folien · Vertraulich*
