# Multivariate Analyse der Einflussfaktoren auf die Heilungsdauer

**Datum:** 2025-03-10

## Übersicht

Diese Analyse baut auf den Ergebnissen der univariaten Analyse auf und untersucht, wie mehrere Faktoren zusammenwirken, um die Heilungsdauer bei Polytrauma-Patienten zu beeinflussen. Es wurden sowohl lineare Regressionsmodelle als auch Überlebensanalysen implementiert.

## Ergebnisse der Regressionsanalyse

### Modell 1: Signifikante Faktoren (Abdomen, Kopf)

- **R²:** 0.419
- **Angepasstes R²:** 0.376
- **Koeffizienten:**
  - const: 608.76 (p=0.0000) *signifikant*
  - Abdomen: 285.51 (p=0.0054) *signifikant*
  - Kopf: -191.96 (p=0.0233) *signifikant*

### Modell 2: Erweitert mit fast signifikanten Faktoren

- **R²:** 0.545
- **Angepasstes R²:** 0.472
- **Koeffizienten:**
  - const: 633.07 (p=0.0000) *signifikant*
  - Abdomen: 339.71 (p=0.0010) *signifikant*
  - Kopf: -110.06 (p=0.1805)
  - Wirbelsaeule: -186.75 (p=0.0383) *signifikant*
  - Arm: 44.09 (p=0.5935)

### Modell 3: Alter mit nicht-linearem Effekt

- **R²:** 0.553
- **Angepasstes R²:** 0.436
- **Koeffizienten:**
  - const: 600.92 (p=0.0229) *signifikant*
  - Abdomen: 321.55 (p=0.0041) *signifikant*
  - Kopf: -102.12 (p=0.2368)
  - Wirbelsaeule: -167.80 (p=0.0858)
  - Arm: 50.85 (p=0.5552)
  - Alter: 2.43 (p=0.8149)
  - Age_Squared: -0.04 (p=0.7241)

## Ergebnisse der Survival-Analyse

### Cox Proportional Hazards Modell

- **Concordance Index:** 0.738
- **Log-Likelihood Ratio Test p-Wert:** 0.0003

- **Hazard Ratios:**
  - Kopf: HR=2.09 (95% KI: 0.81-5.39, p=0.1287)
  - Abdomen: HR=0.16 (95% KI: 0.05-0.52, p=0.0025) *signifikant*
  - Wirbelsaeule: HR=3.68 (95% KI: 1.03-13.14, p=0.0448) *signifikant*
  - Arm: HR=1.08 (95% KI: 0.38-3.07, p=0.8782)

## Wichtigste Erkenntnisse

### Signifikante Einflussfaktoren

- **Abdomen** verlängert die Heilungsdauer um durchschnittlich **339.7 Tage** (p=0.0010), wenn für andere Faktoren kontrolliert wird.
- **Wirbelsaeule** verkürzt die Heilungsdauer um durchschnittlich **186.7 Tage** (p=0.0383), wenn für andere Faktoren kontrolliert wird.

### Modellgüte

- Das beste Modell (model2) erklärt **54.5%** der Varianz in der Heilungsdauer (R²).
- Das angepasste R² beträgt **47.2%**.

### Erkenntnisse aus der Survival-Analyse

Die Cox-Regressionsanalyse identifizierte die folgenden signifikanten Faktoren:

- **Abdomen** verringert die Wahrscheinlichkeit einer längeren Heilungsdauer um den Faktor **6.41** (p=0.0025).
- **Wirbelsaeule** erhöht die Wahrscheinlichkeit einer längeren Heilungsdauer um den Faktor **3.68** (p=0.0448).

## Visualisierungen

Die folgenden Visualisierungen wurden im Rahmen dieser Analyse erstellt:

1. **Regressionskoeffizienten über verschiedene Modelle**: Vergleicht die Stärke und Richtung des Einflusses verschiedener Faktoren.
2. **Tatsächliche vs. Vorhergesagte Heilungsdauer**: Zeigt die Vorhersagekraft des besten Modells.
3. **Kaplan-Meier Kurven nach Verletzungstyp**: Visualisiert den Einfluss verschiedener Verletzungen auf die Heilungswahrscheinlichkeit im Zeitverlauf.
4. **Cox Modell Hazard Ratios**: Stellt das relative Risiko für längere Heilungsdauer dar.

## Vergleich mit univariater Analyse

Die multivariate Analyse baut auf den Ergebnissen der univariaten Analyse auf, die signifikante Einflüsse von Verletzungen des Kopfes und des Abdomens identifiziert hatte. In der multivariaten Analyse zeigt sich, dass...

- Die univariat signifikanten Faktoren **Abdomen** bleiben auch in der multivariaten Analyse signifikant.
- Die Faktoren **Wirbelsaeule** werden erst in der multivariaten Analyse als signifikant erkannt.
- Die univariat signifikanten Faktoren **Kopf** verlieren ihre Signifikanz in der multivariaten Analyse, was auf Konfundierung hindeutet.

## Schlussfolgerungen und Empfehlungen

Basierend auf den Ergebnissen der multivariaten Analyse können die folgenden Schlussfolgerungen gezogen werden:

1. Das multivariate Modell erklärt einen substanziellen Teil der Varianz in der Heilungsdauer (54.5%), was seine klinische Relevanz unterstreicht.
2. Die identifizierten Haupteinflussfaktoren (Abdomen, Wirbelsaeule) sollten besonders bei der Prognose der Heilungsdauer und der Planung von Rehabilitationsmaßnahmen berücksichtigt werden.
3. Die Ergebnisse unterstützen einen individualisierten Ansatz in der Rehabilitation, der die spezifischen Verletzungsmuster jedes Patienten berücksichtigt.

## Limitationen

- **Stichprobengröße**: Die Analyse basiert auf nur 30 Patienten, was die statistische Power einschränkt.
- **Definition der Heilungsdauer**: Die Heilungsdauer wurde als Zeit vom Unfall bis zum letzten Besuch definiert, was möglicherweise nicht immer das tatsächliche Ende des Heilungsprozesses widerspiegelt.
- **Fehlende Variablen**: Möglicherweise wurden wichtige Einflussfaktoren nicht erfasst.
- **Multikollinearität**: Zwischen einigen Verletzungsarten besteht möglicherweise eine Korrelation, die die Interpretation der Koeffizienten erschwert.

## Nächste Schritte

1. **Zeitbasierte Analyse**: Untersuchung des Einflusses des Zeitpunkts von Interventionen auf die Heilungsdauer.
2. **Subgruppenanalyse**: Detaillierte Analyse für bestimmte Patientengruppen (z.B. nach Alter oder Geschlecht).
3. **Entwicklung eines Prognosetools**: Auf Basis der identifizierten Faktoren könnte ein praktisches Tool zur Abschätzung der individuellen Heilungsdauer entwickelt werden.

*Dieser Bericht wurde automatisch im Rahmen der Polytrauma-Analyse erstellt.*