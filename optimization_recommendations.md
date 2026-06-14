# Ottimizzazione I/O CUDA per MTP su Tesla P40

## Introduzione
Questo documento raccoglie le migliori pratiche per ottimizzare il trasferimento host‑device (CPU ↔ GPU) nel contesto del decodifica speculativo MTP (Multi‑Token Prediction) su GPU NVIDIA Tesla P40 (architettura Pascal, compute capability 6.1). Le tecniche descritte mirano a ridurre la latenza e aumentare la banda effettiva di trasferimento, sfruttando le caratteristiche hardware della P40.

---

## 1. Pinned (Page‑Locked) Memory
- **Perché**: Le trasferimenti da/verso memoria paginabile standard richiedono un bounce buffer interno del driver, aggiungendo latenza e consumando banda extra.
- **Come**: Allocare i buffer host con `cudaHostAlloc` (flag `cudaHostAllocDefault` o `cudaHostAllocPortable`) oppure con `cudaMallocHost`. 
- **Vantaggi**:
  - Accesso diretto da parte della DMA della GPU senza intervento della CPU.
  - Possibilità di sovrapporre calcolo e trasferimento tramite stream.
- **Considerazioni sulla P40**:
  - La P40 dispone di 24 GB GDDR5X; l'uso di pinned memory riduce il sovraccarico di copia particolarmente utile quando si trasferiscono blocchi di attivazione/pesi grandi tipici del MTP.
  - Limitare la quantità di pinned memory per evitare di compromettere le prestazioni di sistema (la memoria pinned è riservata e non paginabile).

---

## 2. Memcpy Asincrono
- **Perché**: Le chiamate sincrone `cudaMemcpy` bloccano il thread host finché il trasferimento non termina, impedendo l'overlap con il calcolo.
- **Come**: Utilizzare `cudaMemcpyAsync` specificando uno stream diverso dallo stream di calcolo predefinito (stream 0).
- **Best practice**:
  - Creare almeno due stream: uno per il trasferimento H2D/D2H e uno per il kernel MTP.
  - Eseguire il trasferimento del batch successivo mentre il kernel elabora il batch corrente (double buffering).
  - Sincronizzare esplicitamente solo quando necessario (`cudaStreamSynchronize` o eventi).
- **Esempio**:
  ```cuda
  cudaStream_t computeStream, transferStream;
  cudaStreamCreate(&computeStream);
  cudaStreamCreate(&transferStream);

  // Allocazione pinned host buffer
  float *hostBuf;
  cudaMallocHost(&hostBuf, size);

  // Loop di elaborazione
  for (int i = 0; i < nBatches; ++i) {
      // Trasferimento asincrono del batch i
      cudaMemcpyAsync(d_input, hostBuf + i*size, size, cudaMemcpyHostToDevice, transferStream);
      
      // Lancio kernel su computeStream (attende il completamento del trasferimento tramite dipendenza di stream)
      kernelMTP<<<grid, block, 0, computeStream>>>(d_input, d_output, ...);
      
      // Trasferimento risultato asincrono
      cudaMemcpyAsync(hostBuf + i*size, d_output, size, cudaMemcpyDeviceToHost, transferStream);
  }
  
  cudaStreamSynchronize(transferStream);
  ```

---

## 3. Stream Pipelining (Doppio/Triplo Buffering)
- **Perché**: Massimizzare l'utilizzo della GPU sovrapponendo più fasi (trasferimento H2D, calcolo, trasferimento D2H).
- **Come**: Utilizzare più buffer host/device e più stream in modo che ogni fase lavori su un diverso batch nello stesso momento.
- **Schema a 3 fasi (triplo buffering)**:
  1. Trasferimento H2D del batch N+1
  2. Calcolo kernel sul batch N
  3. Trasferimento D2H del batch N‑1
- **Implementazione**:
  - Mantenere un array di buffer host (pinned) e device.
  - Avanzare gli indici in modo circolare.
  - Usare eventi per garantire le dipendenze tra stream senza blocchi eccessivi.
- **Beneficio sulla P40**: La P40 ha un singolo motore di copia e un singolo motore di calcolo; lo stream pipelining permette di tenere entrambi occupati il più possibile, riducendo il tempo morto dovuto alla serializzazione.

---

## 4. Allineamento VRAM e Coalescenza degli Accessi
- **Perché**: Gli accessi alla memoria globale della GPU sono più efficienti quando sono allineati a 128 byte (segemento di cache L2) e quando i thread di un warp accedono a locazioni contigue.
- **Come**:
  - Allocare i buffer device con un allineamento di almeno 256 byte (usare `cudaMalloc` con padding o `cudaMemAlloc` con allineamento personalizzato).
  - Assicurarsi che le dimensioni dei trasferimenti siano multiple di 128 byte (idealmente 256 byte o più) per sfruttare al massimo le transazioni di memoria GDDR5X.
  - Quando si utilizzano strutture dati (es. tensori di attivazione), prevedere padding alla fine di ogni riga o slice per raggiungere l'allineamento desiderato.
- **Specifiche P40**:
  - Larghezza di banda teorica GDDR5X: ~346 GB/s.
  - Dimensione della linea di cache L2: 128 byte.
  - Allineare a 256 byte garantisce che ogni transazione di memoria sia completamente utilizzata senza split.

---

## 5. Raccomandazioni Specifiche per la Tesla P40 (Pascal, sm_61)
- **Motore di copia**: La P40 dispone di un singolo motore di copia DMA; pertanto, sovrapporre più trasferimenti contemporaneamente non aumenta la banda totale, ma permette di nascondere la latenza di copia dietro il calcolo.
- **Motore di calcolo**: Un singolo set di SM (24 SM, 2048 core CUDA). Utilizzare stream diversi permette al motore di calcolo di rimanere occupato mentre il motore di copia trasferisce il prossimo batch.
- **Clock e banda**: La memoria GDDR5X opera a circa 7 Gbps per pin; con bus a 384 bit si ottiene la banda sopra indicata. Evitare trasferimenti piccoli e frammentati.
- **Temperatura**: Monitorare la temperatura (soglia di throttle ~89 °C). Un utilizzo intenso di copia e calcolo può aumentare il calore; assicurare un adeguato flusso d'aria.
- **Driver e Toolkit**: Utilizzare CUDA Toolkit 11.x o superiore (compatibile con sm_61) per accedere alle ultime ottimizzazioni di `cudaMemcpyAsync` e alla gestione migliorata degli stream.

---

## 6. Esempio di Schema di Lavoro MTP Ottimizzato
```text
+-------------------+    +-------------------+    +-------------------+
| Stream Trasferimento (H2D) | --> | Stream Calcolo (Kernel MTP) | --> | Stream Trasferimento (D2H) |
+-------------------+    +-------------------+    +-------------------+
        ^                         ^                         ^
        |                         |                         |
   Buffer Host (pinned)   Buffer Device   Buffer Host (pinned)
        |                         |                         |
        +-----------+-----------+-----------+
                    |
              Doppio/Triplo Buffering
```
- **Passi**:
  1. Allocare N buffer host pinned e N buffer device (N≥2 per doppio buffering, N≥3 per triplo).
  2. In ogni iterazione:
     - Avviare `cudaMemcpyAsync` dallo host buffer i al device buffer i su stream di trasferimento.
     - Lanciare il kernel MTP sul device buffer i su stream di calcolo (dipende dal completamento della copia tramite evento o stream wait).
     - Avviare `cudaMemcpyAsync` dal device buffer i allo host buffer i su stream di trasferimento (per risultati).
  3. Dopo il loop, sincronizzare lo stream di trasferimento per assicurare il completamento di tutte le operazioni.

---

## 7. Strumenti di Verifica e Profiling
- **Nsight Systems / Nsight Compute**: Per visualizzare l'overlap tra copie e kernel, e identificare eventuali gap.
- **CUDA Events**: Misurare il tempo di ciascuna fase (H2D, kernel, D2H) e calcolare l'effettiva banda raggiunta.
- **metriche di banda**: `bandwidth = (bytes trasferiti) / (tempo misurato)`. Confrontare con la banda teorica della P40 (~346 GB/s) per valutare l'efficienza.

---

## 8. Riferimenti
- NVIDIA CUDA C Programming Guide, sezione "Asynchronous Concurrent Execution" e "Page-Locked Host Memory".
- NVIDIA Tesla P40 Whitepaper (architettura Pascal, specifiche GDDR5X).
- Blog post: "Overlapping Data Transfers with Kernel Execution on Pascal GPUs".
- Documentazione llama.cpp backend CUDA (per eventuali hook di integrazione).

---
*Documento generato per la fase 3/5 del progetto "Ottimizzazione I/O CUDA per MTP su P40".*