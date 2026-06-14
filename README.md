# Ottimizzazione I/O CUDA per MTP su Tesla P40

## Descrizione
Profilatura e ottimizzazione del backend CUDA di llama.cpp per il decodifica speculativo MTP (Multi‑Token Prediction) del modello Qwen3.6-35B-A3B su GPU NVIDIA Tesla P40 (Pascal, compute capability 6.1). L'obiettivo è ridurre i colli di bottiglia nel trasferimento host‑device e migliorare l'allineamento della memoria VRAM, aumentando i tok/s senza incrementare il consumo energetico.

## Architettura
- **Backend**: BeeLLama (turbo3_tcq) su Tesla P40 (CUDA1)
- **Modello**: Qwen3.6-35B-A3B-UD-IQ4_XS.gguf (ctx 131072)
- **Script di profiling**: `profile_mtp.sh`
- **Raccomandazioni**: `optimization_recommendations.md`
- **Template kernel CUDA**: `kernel_template.cu`
- **Configurazione NCCL**: `config_nccl.yaml`
- **Analisi dei risultati**: `analysis.txt`

## Installazione
1. Clonare il repository llama.cpp con supporto BeeLLama.
2. Compilare il backend per compute capability 6.1 (Pascal).
3. Posizionare il modello GGUF nella directory dei modelli.
4. Assicurarsi che le variabili d'ambiente puntino alla GPU corretta (CUDA1=Tesla P40).
5. Copiare i file di questo progetto nella directory di lavoro.

## Uso
- Eseguire lo script di profiling: `./profile_mtp.sh`
- Analizzare l'output in `analysis.txt` e applicare le raccomandazioni in `optimization_recommendations.md`.
- Per testare kernel custom, utilizzare `kernel_template.cu` come punto di partenza.
- Per configurazioni multi‑GPU (se presenti), adattare `config_nccl.yaml`.

## Esempi
```bash
# Avvio profilatura su GPU libera
./profile_mtp.sh

# Visualizzare le raccomandazioni
cat optimization_recommendations.md

# Avviare il modello con ottimizzazioni applicate (esempio generico)
./llama-server -m /path/to/model.gguf -c 131072 -ngl 35 --backend beellama
```

## Stato
✅ COMPLETATO — 2026-06-14
- Mappatura backend attuale completata
- Script di profiling manuale creato e testato
- Raccomandazioni per trasferimento host‑device redatte
- Configurazione NCCL e kernel template prodotti
- Documentazione nel vault di sistema aggiornata