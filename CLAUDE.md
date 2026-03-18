# SRS Backtester

**Langue :** réponds toujours en français.

## Purpose

Backtester pour la stratégie SRS (Session Range Strategy) sur les futures
micro E-mini MES et MNQ. L'objectif est de valider l'edge statistique de
la stratégie et d'explorer/optimiser ses paramètres.

## Strategy Rules — SRS (Session Range Strategy)

### Sessions
- **Session 1** : 9:00 AM Eastern Time (New York)
- **Session 2** : 12:00 PM Eastern Time
- Attention au DST : ET = UTC-5 (EST, hiver) / UTC-4 (EDT, été)

### Logique d'entrée
1. À l'heure de session, enregistrer le high et le low des **30 premières minutes** (range)
2. Attendre un **breakout** au-dessus du high (long) ou en dessous du low (short)
3. Entrée au marché au moment du breakout

### Gestion de position
- **Stop-loss** : côté opposé du range (short → high du range, long → low du range)
- **Take-profit** : basé sur un Risk/Reward ratio (défaut : 2.25)
- Le range en ticks = distance entre high et low du range

### Règles de sortie
1. Take-profit atteint
2. Stop-loss atteint
3. Le prix **clôture à l'intérieur du range** sur une bougie 15 minutes
4. Fin de session (exit forcée)

### Instruments
- **MES** (Micro E-mini S&P 500) : tick size = 0.25, tick value = $1.25
- **MNQ** (Micro E-mini Nasdaq 100) : tick size = 0.25, tick value = $0.50

## Data

### Source
- **Databento** — CME Globex MDP 3.0 (GLBX.MDP3)
- Schema : OHLCV-1m
- Format : DBN (Databento Binary Encoding), compressé zstd
- Période : 2010-06-06 à 2026-03-15 (MES/MNQ existent depuis mai 2019)

### Fichiers
- `data/raw/` : fichiers DBN bruts (non versionnés, dans .gitignore)
- `data/processed/` : données nettoyées en Parquet (non versionnées)

## Backtesting Methodology

### Biais à éviter
- **Look-ahead bias** : toujours utiliser shift(1) ou vérifier qu'on ne regarde
  pas dans le futur
- **Survivorship bias** : pas applicable ici (futures, pas d'actions)
- **Overfitting** : split train/test/validation, walk-forward analysis

### Contract rollover
- Les données contiennent tous les contrats trimestriels
- Utiliser le front-month contract et gérer le roll proprement
- Vérifier que le roll ne crée pas de faux signaux aux dates de transition

### Coûts réalistes
- **Commissions** : modéliser les frais broker (NinjaTrader fees)
- **Slippage** : 1 tick par trade par défaut (conservateur pour micros)

### Métriques à calculer
- Win rate
- Profit factor
- Max drawdown
- Sharpe ratio
- Nombre total de trades
- PnL moyen par trade
- Performance par session (session 1 vs session 2)
- Performance par instrument (MES vs MNQ)
- Distribution des trades par jour de la semaine

## Tech Stack
- Python 3.10+
- `databento` — lecture des fichiers DBN
- `pandas` — manipulation de données
- `numpy` — calculs numériques
- Backtester custom (pandas-based) pour commencer

## File Structure
```
srs-backtester/
├── CLAUDE.md              ← ce fichier
├── .gitignore
├── requirements.txt
├── data/
│   ├── raw/               ← fichiers DBN Databento (gitignored)
│   └── processed/         ← Parquet nettoyés (gitignored)
├── src/
│   ├── data_loader.py     ← chargement et nettoyage des données
│   ├── strategy.py        ← logique SRS
│   ├── backtester.py      ← moteur de backtest
│   └── metrics.py         ← calcul des métriques
├── notebooks/             ← exploration et visualisation
└── results/               ← résultats de backtest (gitignored)
```
