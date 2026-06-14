#!/usr/bin/env bash
# profile_mtp.sh - Script per profilare il decoding MTP di Qwen3.6-35B-A3B su GPU P40
# Utilizza Nsight Systems (nsys) per raccogliere metriche di performance.
# Rispetta le vincoli di privacy: nessun path assoluto hardcoded, nessun dato personale.

set -euo pipefail

# ======================
# Configurazione (override tramite variabili d'ambiente)
# ======================
: "${MODEL_PATH:?Errore: impostare MODEL_PATH sul percorso del modello Qwen3.6-35B-A3B}"
: "${PRESET_NAME:=}"          # Nome preset start-llama (opzionale)
: "${PORT:=8090}"             # Porta del server llama
: "${BINARY:=beellama}"       # Binario da usare (beellama per P40)
: "${CTX_SIZE:=131072}"       # Dimensione context window
: "${NGL:=999}"               # Numero di layer da offloadare su GPU
: "${MTP_FLAG:="--mtp"}"     # Flag per abilitare MTP (regolare se necessario)
: "${PROFILING_TOOL:=nsys}"   # Strumento di profilazione (nsys o nvprof)
: "${PROFILING_DURATION:=30}" # Secondi di profilazione
: "${OUTPUT_DIR:=./profile_results}" # Directory per risultati
: "${REQUEST_PROMPT:=\"Ciao, come stai?\"}" # Prompt per test di generazione
: "${MAX_TOKENS:=128}"        # Token da generare per il test

# ======================
# Funzioni di supporto
# ======================
log() {
    echo "[profile_mtp] $*"
}

error() {
    log "ERRORE: $*" >&2
    exit 1
}

check_command() {
    command -v "$1" >/dev/null 2>&1 || error "Comando richiesto non trovato: $1"
}

wait_for_server() {
    local max_attempts=30
    local attempt=1
    while (( attempt <= max_attempts )); do
        if curl -s http://127.0.0.1:${PORT}/health | grep -q '"status":"ok"'; then
            log "Server pronto sulla porta ${PORT}"
            return 0
        fi
        sleep 1
        ((attempt++))
    done
    error "Timeout nell'attesa del server sulla porta ${PORT}"
}

# ======================
# Verifiche iniziali
# ======================
log "Avvio script di profilazione MTP"

# Verifica presenza strumenti richiesti
check_command "curl"
check_command "${PROFILING_TOOL}"
check_command "start-llama"  # nello PATH dell'utente

# Verifica esistenza modello
if [[ ! -f "${MODEL_PATH}" ]]; then
    error "File modello non trovato: ${MODEL_PATH}"
fi

# Crea directory output
mkdir -p "${OUTPUT_DIR}"
log "Directory risultati: ${OUTPUT_DIR}"

# ======================
# Avvio server llama
# ======================
log "Avvio server llama con preset '${PRESET_NAME:-default}' sulla porta ${PORT}"

# Costruisce comando start-llama
CMD=(start-llama)
if [[ -n "${PRESET_NAME}" ]]; then
    CMD+=("${PRESET_NAME}")
fi
CMD+=(
    --binary "${BINARY}"
    --model "${MODEL_PATH}"
    --port "${PORT}"
    --ctx "${CTX_SIZE}"
    --ngl "${NGL}"
    ${MTP_FLAG}
    --dry-run  # Prima facciamo un dry-run per vedere il comando
)

log "Comando dry-run: ${CMD[*]}"
"${CMD[@]}"  # Esegue dry-run e mostra il comando reale

# Ora avvia per reale (senza --dry-run)
CMD_NO_DRY=()
for arg in "${CMD[@]}"; do
    if [[ "${arg}" == "--dry-run" ]]; then
        continue
    fi
    CMD_NO_DRY+=("${arg}")
done

log "Avvio server in background..."
"${CMD_NO_DRY[@]}" &
SERVER_PID=$!
log "Server avviato con PID ${SERVER_PID}"

# Attendi che il server sia pronto
wait_for_server

# ======================
# Profilazione con nsys/nvprof
# ======================
log "Avvio profilazione con ${PROFILING_TOOL} per ${PROFILING_DURATION}s"

# Prepara comando di profilazione
if [[ "${PROFILING_TOOL}" == "nsys" ]]; then
    PROF_CMD=(
        nsys profile
        --output="${OUTPUT_DIR}/nsys_report"
        --trace=cuda,nvtx,osrt
        --cpuctxsw=true
        --gputrace=true
        --demangle=true
        --force-overwrite true
        --pid="${SERVER_PID}"
    )
elif [[ "${PROFILING_TOOL}" == "nvprof" ]]; then
    PROF_CMD=(
        nvprof
        --output-profile "${OUTPUT_DIR}/nvprof_report.nvvp"
        --csv
        --log-file "${OUTPUT_DIR}/nvprof_log.txt"
        --profile-from-start off
        -i "${SERVER_PID}"
    )
else
    error "Strumento di profilazione non supportato: ${PROFILING_TOOL}"
fi

log "Comando profilazione: ${PROF_CMD[*]}"

# Avvia profilazione in background
"${PROF_CMD[@]}" &
PROF_PID=$!
log "Profilazione avviata con PID ${PROF_PID}"

# Attendi un momento per assicurarsi che la profilazione sia iniziata
sleep 2

# ======================
# Esegui test di generazione
# ======================
log "Invio richiesta di generazione al server..."
RESPONSE=$(curl -s -X POST http://127.0.0.1:${PORT}/completion \
    -H "Content-Type: application/json" \
    -d "{
        \"prompt\": \"${REQUEST_PROMPT}\",
        \"n_predict\": ${MAX_TOKENS},
        \"temperature\": 0.8,
        \"top_p\": 0.95,
        \"repeat_penalty\": 1.1
    }" || true)

if [[ -z "${RESPONSE}" ]]; then
    log "Avviso: nessuna risposta dal server (potrebbe essere ancora in profilazione)"
else
    log "Risposta ricevuta (primi 100 caratteri): ${RESPONSE:0:100}"
fi

# ======================
# Fermare profilazione e server
# ======================
log "Attesa completamento profilazione (${PROFILING_DURATION}s)..."
sleep "${PROFILING_DURATION}"

log "Fermando profilazione..."
if [[ "${PROFILING_TOOL}" == "nsys" ]]; then
    # nsys termina quando il processo target termina o dopo il tempo specificato
    # Inviamo SIGINT per terminare pulitamente se necessario
    kill -INT "${PROF_PID}" 2>/dev/null || true
elif [[ "${PROFILING_TOOL}" == "nvprof" ]]; then
    kill -INT "${PROF_PID}" 2>/dev/null || true
fi
wait "${PROF_PID}" 2>/dev/null || true

log "Fermando server llama..."
kill "${SERVER_PID}" 2>/dev/null || true
wait "${SERVER_PID}" 2>/dev/null || true

log "Profilazione completata. Risultati in: ${OUTPUT_DIR}"
log "Per visualizzare il report nsys: nsys ui ${OUTPUT_DIR}/nsys_report.nsys-rep"
if [[ -f "${OUTPUT_DIR}/nsys_report.sqlite" ]]; then
    log " oppure: sqlite3 ${OUTPUT_DIR}/nsys_report.sqlite '.schema'"
fi

exit 0