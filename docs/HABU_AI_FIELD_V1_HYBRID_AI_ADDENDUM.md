# Habu AI Field v1 Hybrid AI Addendum

## Decision

Habu AI Field v1 shall support a hybrid prediction architecture that combines:

1. dedicated predictive machine-learning models trained on measured Habu field data; and
2. a generative-AI reasoning layer that acts as a Habu-domain specialist.

The generative-AI layer is optional in the scoring path and must only affect production predictions when empirical validation shows that it improves the defined prediction KPIs. Product appearance or conversational quality must never override measured predictive performance.

## Roles

### Predictive ML engine
The predictive ML engine remains the numeric source of truth for measurable predictions such as:
- nightly capture-count forecast
- 1 km area ranking
- 500/250/100 m road-zone probability
- 10 m road segment x 10 min scores
- large-Habu probability
- route expected captures / efficiency

It learns from canonical field data including capture, sighting, GPX exposure/non-capture observations, road/environmental features, time, weather, season and validated derived features.

### Generative-AI Habu specialist
The generative-AI layer acts as a domain reasoning and synthesis module. It may consume:
- outputs and uncertainty from predictive models
- current/cached weather
- road and route state
- recent field observations
- historical Habu research and extracted structured findings
- verified biological/ecological domain knowledge
- model/version/performance metadata

It may:
- explain why the model favors an area or time window
- identify unusual combinations or distribution shift
- suggest candidate derived features or hypotheses for later validation
- compare routes and risks in natural language
- produce an operation briefing from the numeric forecast
- flag conflicts between measured data and domain expectations

It must not:
- invent coordinates or road geometry
- overwrite immutable pre-search forecasts after exploration begins
- fabricate weather, field observations or research facts
- convert road-event discovery time into actual presence/activity time
- silently change a numeric prediction merely because a language-model narrative sounds plausible

## Accuracy-first model competition

Production shall evaluate at least these predictor modes:

- `ML_ONLY`: dedicated predictive model only
- `ML_PLUS_GENAI`: predictive model plus generative-AI-derived structured signals / adjudication
- optional `ENSEMBLE`: multiple dedicated predictive models combined, with or without generative-AI signals

Each mode shall be scored on held-out/backtest nights using the same frozen prediction protocol.

Primary metrics include:
- nightly capture-count error
- 1 km Area Top10 Recall
- official 100 m ±20 min KPI
- auxiliary 50 m ±10 min, 100 m ±10 min, 100 m ±30 min, 250 m ±30 min
- route expected-vs-actual capture performance

The production mode is selected by measured out-of-sample performance, not by preference for a particular AI technology.

If generative AI fails to improve predictive performance, ML_ONLY remains the production predictor while generative AI may still provide explanation, research synthesis and user interaction.

## Structured generative-AI interface

The generative-AI layer must consume and emit versioned structured data. Numeric forecast fields must be traceable to their originating model/version. Free-form text must not be parsed back into critical prediction values when a structured field can be used.

Suggested structured output fields:
- `domain_flags[]`
- `distribution_shift_flags[]`
- `research_consistency_notes[]`
- `candidate_feature_hypotheses[]`
- `route_interpretation[]`
- `confidence_adjustment_proposal` (proposal only unless validated model contract explicitly permits it)
- `human_briefing`

## Specialist knowledge base

The generative-AI specialist should be grounded on a curated Habu knowledge base built from validated sources and the project's own measured data. Research claims should be stored with source/provenance and, where possible, structured numeric findings rather than only prose summaries.

The specialist must distinguish:
- measured project data
- published research evidence
- model inference
- hypothesis/speculation

## Offline behavior

The field app must remain usable offline. Dedicated on-device predictive inference, map, GPS, route execution and field logging must not depend on a live generative-AI connection.

When generative AI is unavailable:
- use the best validated local predictive mode
- show the numeric forecast and routes normally
- mark specialist commentary as unavailable/stale rather than blocking exploration

## Learning policy

Generative-AI suggestions do not become model features automatically. New hypotheses/features must pass the normal feature engineering, leakage checks, backtest and acceptance criteria before entering a production predictive model.

## User experience

A user asking `今日の予想は？` should receive one coherent answer assembled from the same versioned prediction artifact used by the app. The answer should prioritize:
1. point capture forecast
2. fixed time windows
3. top three routes
4. concise domain reasoning
5. uncertainty / stale-data warnings

The conversational layer must never contradict the frozen numeric artifact without clearly identifying that it is proposing an alternative hypothesis rather than reporting the official forecast.

## Acceptance criteria

Hybrid AI is considered successfully integrated when:
- ML_ONLY and ML_PLUS_GENAI can be evaluated side-by-side on identical historical nights
- mode/version/provenance are recorded for every official forecast
- production selects the empirically superior mode under frozen metrics
- generative AI cannot invent or mutate GIS geometry
- offline field operation works without generative AI
- specialist explanations identify whether statements come from project data, published research, model inference or hypothesis
