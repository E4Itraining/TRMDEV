# Lab — Observabilité ML avec Phoenix (Arize)

## Objectif

Apprendre à instrumenter un pipeline d'entraînement ML avec **Phoenix by Arize**
et **OpenTelemetry** pour obtenir une observabilité complète : traces, métriques,
et comparaison d'expériences.

Le use case : le **Tiny Recursive Model (TRM)**, un réseau de neurones récursif
à profondeur adaptative. Le modèle décide dynamiquement combien de fois recurser
pour chaque échantillon — un comportement idéal à observer et analyser.

## Prérequis

- Python 3.10+
- Les dépendances du projet TRM (PyTorch, PyYAML)

## Installation

```bash
# Depuis la racine du projet
pip install -r labs/observability-phoenix/requirements.txt
```

## Architecture du tracing

```
                    ┌─────────────────────┐
                    │   Phoenix UI        │
                    │   localhost:6006     │
                    └─────────┬───────────┘
                              │ OTLP/HTTP
                              │
                    ┌─────────┴───────────┐
                    │  OpenTelemetry SDK  │
                    │  TracerProvider     │
                    └─────────┬───────────┘
                              │
              ┌───────────────┼───────────────┐
              │               │               │
        ┌─────┴─────┐  ┌─────┴─────┐  ┌──────┴─────┐
        │  Epoch    │  │  Step     │  │ Validation │
        │  Span     │  │  Span     │  │ Span       │
        └───────────┘  └───────────┘  └────────────┘
```

Chaque span porte des **attributs riches** :

| Span             | Attributs clés                                       |
|------------------|------------------------------------------------------|
| `training_run`   | model.params, training.epochs, device                |
| `epoch`          | epoch.number, avg_loss, avg_accuracy, duration_s     |
| `training_step`  | loss, accuracy, mean_depth, min/max depth, lr        |
| `validation`     | val.loss, val.accuracy, val.num_samples              |

## Les 3 exercices

### Lab 01 — Phoenix Quickstart

> Se familiariser avec Phoenix et le tracing OpenTelemetry

```bash
cd /home/user/TRMDEV
python labs/observability-phoenix/01_phoenix_quickstart.py
```

**Ce que vous apprendrez :**
- Lancer Phoenix en local
- Configurer un `TracerProvider` OpenTelemetry
- Créer des spans manuels avec attributs
- Naviguer dans l'UI Phoenix (onglet Traces)

**Points d'exploration dans l'UI :**
- Ouvrez `http://localhost:6006`
- Onglet **Traces** → cliquez sur `pipeline.ml`
- Observez la hiérarchie parent-enfant des spans
- Cliquez sur chaque span pour voir ses attributs

---

### Lab 02 — Entraînement instrumenté

> Instrumenter la vraie boucle d'entraînement du TRM

```bash
cd /home/user/TRMDEV
python labs/observability-phoenix/02_traced_training.py
```

Ou avec la config par défaut (plus long) :
```bash
python labs/observability-phoenix/02_traced_training.py --config configs/default.yaml
```

**Ce que vous apprendrez :**
- Instrumenter une boucle d'entraînement PyTorch
- Capturer loss, accuracy, profondeur de récursion comme attributs de span
- Visualiser la progression de l'entraînement dans Phoenix
- Identifier les steps lents ou les anomalies

**Points d'exploration dans l'UI :**
- Filtrez les spans par type (`training_step`, `validation`, `epoch`)
- Triez les `training_step` par `step.mean_depth` pour trouver les
  échantillons les plus complexes
- Comparez la durée des epochs
- Observez l'évolution de la loss dans les attributs

---

### Lab 03 — Comparaison d'expériences

> Comparer l'effet d'un hyperparamètre sur le comportement du modèle

```bash
cd /home/user/TRMDEV
python labs/observability-phoenix/03_experiments.py
```

**Ce que vous apprendrez :**
- Créer des tracers séparés par expérience (projets Phoenix distincts)
- Comparer des runs avec des hyperparamètres différents
- Analyser l'impact du `gate_threshold` sur la profondeur de récursion

**Les 3 expériences :**

| Expérience | `gate_threshold` | Comportement attendu         |
|------------|------------------|------------------------------|
| `deep`     | 0.3              | Récursion profonde           |
| `default`  | 0.5              | Standard                     |
| `shallow`  | 0.7              | Récursion courte, early-stop |

**Points d'exploration dans l'UI :**
- Comparez les projets `trm-experiment-deep`, `default`, `shallow`
- Pour chaque expérience, regardez `step.mean_depth`
- Question : un modèle plus profond est-il plus précis ?

## Concepts clés

### Phoenix by Arize

Phoenix est une plateforme d'observabilité open-source pour l'IA/ML.
Contrairement aux outils classiques (Weights & Biases, MLflow), Phoenix
se base sur **OpenTelemetry**, le standard ouvert du tracing distribué.

### OpenTelemetry (OTel)

Standard CNCF pour l'observabilité. Les concepts clés :

- **Trace** : un parcours complet (ex: un entraînement entier)
- **Span** : une unité de travail (ex: un step d'entraînement)
- **Attributs** : métadonnées attachées à un span (ex: loss=0.42)
- **TracerProvider** : configure où envoyer les traces
- **Exporter** : envoie les traces vers un backend (ici Phoenix via OTLP)

### Pourquoi c'est utile pour le ML ?

1. **Debugging** : identifier pourquoi un training diverge
2. **Profiling** : trouver les bottlenecks (data loading, forward pass)
3. **Comparaison** : A/B testing d'hyperparamètres
4. **Production** : surveiller un modèle déployé en temps réel
5. **Reproductibilité** : tracer exactement ce qui s'est passé

## Structure des fichiers

```
labs/observability-phoenix/
├── README.md                  ← vous êtes ici
├── requirements.txt           ← dépendances Phoenix + OTel
├── lab_config.yaml            ← config courte pour les labs
├── 01_phoenix_quickstart.py   ← Lab 01 : découverte Phoenix
├── 02_traced_training.py      ← Lab 02 : training instrumenté
└── 03_experiments.py          ← Lab 03 : comparaison d'expériences
```

## Pour aller plus loin

- Ajouter des **événements** (`span.add_event()`) pour logger les checkpoints
- Instrumenter le **data loading** avec des spans dédiés
- Utiliser les **métriques OTel** (counters, histogrammes) en plus des traces
- Connecter Phoenix à **Arize Cloud** pour du monitoring en production
- Ajouter de l'**auto-instrumentation** pour les appels HTTP/DB
