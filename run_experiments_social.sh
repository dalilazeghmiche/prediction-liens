#!/usr/bin/env bash
# Automatisation des experiences de l'article [2] (hep-th).
# Lancement : bash run_experiments_social.sh [--quick | --all | --tests-only | ...]
set -Eeuo pipefail

usage() {
  cat <<'HELP'
Usage : bash run_experiments_social.sh [MODE] [OPTIONS]

Modes (un seul ; --all par defaut) :
  --all               Tests + protocole + audit + CN/AA + RWR + courbes
                      + comparaison RWR + benchmarks + validation des CSV.
  --quick             Tests + controle protocole + evaluation CN/AA.
  --tests-only        Tests unitaires uniquement.
  --audit-only        Controle du protocole + audit du corpus.
  --evaluations-only  Evaluation Top-k CN/AA + RWR.
  --curves-only       Courbes precision-rappel CN/AA/RWR.
  --comparison-only   Comparaison RWR directionnel et symetrise.
  --benchmarks-only   Temps et memoire CN/AA/RWR.
  --validate-only     Controle des CSV generes par une campagne complete.

Options :
  --repeats N         Repetitions des benchmarks (defaut : 3).
  --sizes S           Tailles benchmarks, ex. 500,1000,2000,all.
  --seed N            Graine des sous-graphes benchmarks (defaut : 42).
  --worker-timeout N  Delai max par processus benchmark (defaut : 120 s).
  --alpha X           Probabilite de redemarrage RWR (defaut : 0.15).
  --tol X             Tolerance de convergence RWR (defaut : 1e-10).
  --max-iter N        Nombre maximal d'iterations RWR (defaut : 1000).
  --block-size N      Nombre de sources par bloc RWR (defaut : 64).
  -h, --help          Affiche cette aide.

Interpreter : en priorite PYTHON (variable d'environnement), puis
.venv/Scripts/python.exe (Windows Git Bash), puis .venv/bin/python,
puis python3/python.

Sorties : results/<experience>/ ; journaux : results/logs/<lancement>/.
HELP
}

MODE=all
MODE_GIVEN=0
REPEATS=3
SIZES='500,1000,2000,all'
SEED=42
WORKER_TIMEOUT=120
ALPHA='0.15'
TOL='1e-10'
MAX_ITER=1000
BLOCK_SIZE=64

set_mode() {
  if (( MODE_GIVEN )); then
    echo "Erreur : choisir un seul mode (actuel : --$MODE ; nouveau : --$1)." >&2
    exit 2
  fi
  MODE="$1"
  MODE_GIVEN=1
}

while (($#)); do
  case "$1" in
    --all) set_mode all; shift ;;
    --quick) set_mode quick; shift ;;
    --tests-only) set_mode tests; shift ;;
    --audit-only) set_mode audit; shift ;;
    --evaluations-only) set_mode evaluations; shift ;;
    --curves-only) set_mode curves; shift ;;
    --comparison-only) set_mode comparison; shift ;;
    --benchmarks-only) set_mode benchmarks; shift ;;
    --validate-only) set_mode validate; shift ;;
    --repeats|--sizes|--seed|--worker-timeout|--alpha|--tol|--max-iter|--block-size)
      option="$1"; shift
      if (($# == 0)); then echo "Valeur manquante pour $option" >&2; exit 2; fi
      case "$option" in
        --repeats) REPEATS="$1" ;;
        --sizes) SIZES="$1" ;;
        --seed) SEED="$1" ;;
        --worker-timeout) WORKER_TIMEOUT="$1" ;;
        --alpha) ALPHA="$1" ;;
        --tol) TOL="$1" ;;
        --max-iter) MAX_ITER="$1" ;;
        --block-size) BLOCK_SIZE="$1" ;;
      esac
      shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Option inconnue : $1" >&2; usage >&2; exit 2 ;;
  esac
done

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

if [[ -n "${PYTHON:-}" ]]; then
  PY="$PYTHON"
elif [[ -f .venv/Scripts/python.exe ]]; then
  PY=".venv/Scripts/python.exe"
elif [[ -x .venv/bin/python ]]; then
  PY=".venv/bin/python"
elif command -v python3 >/dev/null 2>&1; then
  PY="$(command -v python3)"
elif command -v python >/dev/null 2>&1; then
  PY="$(command -v python)"
else
  echo 'Python introuvable. Creez un environnement virtuel ou definissez PYTHON.' >&2
  exit 1
fi

[[ -f requirements.txt ]] || { echo 'requirements.txt introuvable : lancer depuis le depot.' >&2; exit 1; }

case "$MODE" in
  all) REQUIRED=(verifier_protocole_partie1.py audit_hepth_article2.py evaluation_cn_aa_hepth.py evaluation_rwr_hepth.py courbes_precision_rappel_hepth.py comparaison_rwr_original_symetrise_hepth.py benchmarks_cn_aa_rwr_hepth.py verifier_resultats_social.py) ;;
  quick) REQUIRED=(verifier_protocole_partie1.py evaluation_cn_aa_hepth.py) ;;
  audit) REQUIRED=(verifier_protocole_partie1.py audit_hepth_article2.py) ;;
  evaluations) REQUIRED=(evaluation_cn_aa_hepth.py evaluation_rwr_hepth.py) ;;
  curves) REQUIRED=(courbes_precision_rappel_hepth.py) ;;
  comparison) REQUIRED=(comparaison_rwr_original_symetrise_hepth.py) ;;
  benchmarks) REQUIRED=(benchmarks_cn_aa_rwr_hepth.py) ;;
  validate) REQUIRED=(verifier_resultats_social.py) ;;
  tests) REQUIRED=() ;;
esac
for f in "${REQUIRED[@]}"; do
  [[ -f "$f" ]] || { echo "Script manquant : $f (a placer a la racine du depot)" >&2; exit 1; }
done

if [[ "$MODE" != 'tests' && "$MODE" != 'validate' ]]; then
  for file in data/raw/hep-th-abs.tar.gz data/raw/hep-th-slacdates.tar.gz; do
    [[ -f "$file" ]] || { echo "Archive manquante : $file" >&2; exit 1; }
  done
fi

"$PY" -c 'import sys; print("Python :", sys.executable, sys.version.split()[0])'
if [[ "$MODE" != 'tests' && "$MODE" != 'audit' && "$MODE" != 'validate' ]]; then
  "$PY" -c 'import numpy, scipy' || {
    echo 'Dependances manquantes : installer requirements-experiences.txt.' >&2; exit 1;
  }
fi
if [[ "$MODE" == 'all' || "$MODE" == 'curves' || "$MODE" == 'comparison' || "$MODE" == 'benchmarks' ]]; then
  "$PY" -c 'import matplotlib' || {
    echo 'Matplotlib manquant : installer requirements-experiences.txt.' >&2; exit 1;
  }
fi
if [[ "$MODE" == 'all' || "$MODE" == 'benchmarks' ]]; then
  "$PY" -c 'import psutil' || {
    echo 'psutil manquant : installer requirements-experiences.txt.' >&2; exit 1;
  }
fi

RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)_$$"
LOG_DIR="$ROOT/results/logs/$RUN_ID"
mkdir -p "$LOG_DIR"
SUMMARY="$LOG_DIR/manifest.txt"
{
  echo "mode=$MODE"
  echo "utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "python=$PY"
  echo "repeats=$REPEATS sizes=$SIZES seed=$SEED worker_timeout=$WORKER_TIMEOUT"
  echo "alpha=$ALPHA tol=$TOL max_iter=$MAX_ITER block_size=$BLOCK_SIZE"
  if command -v git >/dev/null 2>&1 && git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
    echo "git_commit=$(git rev-parse HEAD)"
    if [[ -n "$(git status --porcelain)" ]]; then echo 'git_clean=no'; else echo 'git_clean=yes'; fi
  fi
  "$PY" -c 'import platform, importlib.metadata as m; print("os="+platform.platform()); print("machine="+platform.machine()); print("processor="+platform.processor()); [print(pkg+"="+m.version(pkg)) for pkg in ("numpy", "scipy", "matplotlib", "psutil") if m.packages_distributions().get(pkg)]' 2>/dev/null || true
} > "$SUMMARY"
"$PY" -m pip freeze > "$LOG_DIR/pip-freeze.txt" 2>/dev/null || true

run_step() {
  local name="$1"; shift
  echo
  echo "================================================================"
  echo "[DEBUT] $name"
  echo "================================================================"
  printf '%q ' "$@" >> "$SUMMARY"; printf '\n' >> "$SUMMARY"
  "$@" 2>&1 | tee "$LOG_DIR/$name.log"
  echo "[OK] $name"
}

RWR=(--alpha "$ALPHA" --tol "$TOL" --max-iter "$MAX_ITER" --block-size "$BLOCK_SIZE")
BENCH=(--repeats "$REPEATS" --sizes "$SIZES" --seed "$SEED" --worker-timeout "$WORKER_TIMEOUT")

case "$MODE" in
  all|quick|tests)
    run_step tests "$PY" -m unittest discover -s tests -v ;;
esac
case "$MODE" in
  all|quick|audit)
    run_step controle_protocole "$PY" verifier_protocole_partie1.py ;;
esac
case "$MODE" in
  all|audit)
    run_step audit_corpus "$PY" audit_hepth_article2.py ;;
esac
case "$MODE" in
  all|quick|evaluations)
    run_step evaluation_cn_aa "$PY" evaluation_cn_aa_hepth.py ;;
esac
case "$MODE" in
  all|evaluations)
    run_step evaluation_rwr "$PY" evaluation_rwr_hepth.py "${RWR[@]}" ;;
esac
case "$MODE" in
  all|curves)
    run_step courbes_precision_rappel "$PY" courbes_precision_rappel_hepth.py "${RWR[@]}" ;;
esac
case "$MODE" in
  all|comparison)
    run_step comparaison_rwr "$PY" comparaison_rwr_original_symetrise_hepth.py "${RWR[@]}" ;;
esac
case "$MODE" in
  all|benchmarks)
    run_step benchmarks "$PY" benchmarks_cn_aa_rwr_hepth.py "${BENCH[@]}" "${RWR[@]}" ;;
esac
case "$MODE" in
  all|validate)
    run_step validation_sorties "$PY" verifier_resultats_social.py ;;
esac

echo
echo '[TERMINE] Tous les blocs demandes se sont executes sans erreur.'
echo "Fichiers : $ROOT/results/"
echo "Logs et manifeste : $LOG_DIR/"
