# Resoconto pre-sviluppo — Timestral Sandbox

Data: 2026-10-27 · Stato: **Fasi 0-3 e Showcase completati (§12); prossima: dataset online (resto della Fase 4)**. Repository: `tsfm-lab` (privato)

Questo documento riassume requisiti, ricerca (modello + identità visiva Mistral), architettura proposta, rischi e piano di lavoro. Ogni dato è marcato come **verificato** (letto da fonte primaria o misurato), **dichiarato dal vendor** o **inferenza/proposta**.

---

## 1. Requisiti (dalla tua richiesta)

| # | Requisito |
|---|---|
| R1 | Oggi solo **Timestral**, ma l'architettura prevede già l'aggiunta di altri modelli per il confronto. |
| R2 | Web app / interfaccia locale: gestione modelli, test su dataset **locali, online e di test inclusi**, con specifiche e **performance misurate in locale**, grafici con ogni dettaglio per trasparenza. |
| R3 | Interfaccia fedele **1:1** a brand identity e design di Mistral, recuperati dal sito ufficiale. |
| R4 | Progetto isolato (`.venv`) senza file sparsi nel PC: rimuovere la cartella = rimuovere tutto. |
| R5 | Compatibile con Windows, Linux e macOS. |

---

## 2. Il modello in sintesi (dettagli completi nella ricerca precedente)

- 4 taglie: 5M (8 layer, d=192), 28M (12, 384), 123M (20, 640), 209M (24, 768). Patch da 32 passi, contesto fino a 32.768, orizzonte 1.024 per passata (oltre: blocchi autoregressivi sulla mediana), 21 quantili (0.01…0.99). **Verificato.**
- Input `(B, V, T)`, NaN = mancante, covariate passate/future/categoriche. **Verificato.**
- Pesi fp32: 21 / 111 / 490 / 837 MB (~1,46 GB in totale). **Verificato.**
- Latenza ufficiale su CPU server a 8 thread (96 passi da 2016): **17 ms (5M), 113 ms (209M)**. **Dichiarato dal vendor**; i 10/140 ms dei blog sono errati.
- Classifiche su GIFT-Eval / TIME / fev-bench: **dichiarate da Mistral**, margini minimi, non confermate in modo indipendente; report tecnico non ancora uscito.
- Licenza: codice Apache-2.0; **pesi "Modified MIT" con esclusione se i ricavi mensili consolidati della tua azienda/datore di lavoro superano 20 M$**, estesa a derivati e "combined works". **Verificato sul testo grezzo.**

### Criticità tecniche note (repo upstream, tutte aperte e senza risposta dei maintainer)
| Issue | Impatto sul progetto | Mitigazione |
|---|---|---|
| #6 Crash su MPS (float64 in `scaler.py`/`forecaster.py`) | Mac con GPU Apple non utilizzabile con il pacchetto originale | Mac: CPU di default; MPS solo come opzione sperimentale futura |
| Memoria quadratica in `serie × (1+covariate)` per chiamata | Una chiamata non spezzata ha tentato 62 GiB | Batching a blocchi con stima preventiva della memoria |
| #7 Quantile 0.99 che esplode con covariate piatte nel contesto + `symmetric_averaging` | Previsioni inaffidabili con prezzi/covariate quasi costanti | Controlli preventivi sulle covariate + preset di mitigazione (mascherare a NaN, `symmetric_averaging=False`) |
| #5 Non è su PyPI; repo non accetta PR | Dipendenza fragile | Installazione da git con **commit SHA bloccato** nel lockfile |
| Calibrazione (test indipendente Dubach, 28M, finanza) | Intervalli al 90% coprivano ~84% | Grafico di calibrazione sempre in evidenza |

---

## 3. Ricerca brand: cosa ho trovato e cosa no

### 3.1 La pagina ufficiale `mistral.ai/brand`
**Contiene** solo: regole d'uso del logo (rispettare lo spazio libero; preferire la versione gradiente, monocromatica su sfondi complessi; non deformare, ruotare, ricolorare, mettere su sfondo colorato o incorniciare; non usare versioni vecchie/non ufficiali), le varianti (lockup completo e icona: gradiente/invertito/nero/bianco), 4 zip di asset (`Mistral_Brandkit_2026`, `_Logos_`, `_Models_`, `_Product_Shots_`) e la frase che partner e collaboratori possono usare il brand "con cura".
**Non contiene** palette con esadecimali, tipografia, regole sulle illustrazioni pixel. Gli zip sono ospitati su un CMS di terze parti e rispondono **403** agli accessi automatici (non ho tentato di aggirarlo). Non so se il brand kit contenga linee guida più complete.

### 3.2 Design system reale estratto dal CSS pubblicato (misura diretta, 2026-10-27)
Fonte: fogli di stile di `mistral.ai` (Tailwind) + stili calcolati nel browser sulla pagina Timestral e screenshot della home.

**Aspetto generale (verificato a vista):** tema scuro di default (`<html class="dark">`), navbar alta 48 px con celle separate da linee sottili, griglia a hairline, illustrazioni pixel-art, icone costruite con quadratini da 4 px, tag in Space Mono maiuscolo.

**Colori — tema scuro (default)**
| Ruolo | Token | Valore |
|---|---|---|
| Sfondo pagina | `surface-brand-primary` (= steel-950) | `#101013` |
| Card / figure | `surface-brand-secondary` (= steel-900) | `#1a1a1e` |
| Superficie terziaria | steel-800 | `#27272b` |
| Bordo | `border-primary` (= steel-800) | `#27272b` |
| Bordo secondario | `border-secondary` | `#474754` |
| Testo primario | cream-50 | `#fafaf4` |
| Testo attenuato | zinc-400 | `#a1a1aa` |
| Accento brand | `surface-brand` = orange-500 | `#ff5229` |
| Arancio di marca | orange-600 | `#fa500f` |
| Tangerine / giallo / giallo chiaro | tangerine-500 / yellow-500 / yellow-300 | `#ff8204` / `#ffaf01` / `#ffdd73` |

**Colori — tema chiaro:** sfondo `#fbfbf8`, secondario `#f5f4ef`, terziario `#ebe9e0`, bordo `#e4e3de`, bordo secondario `#c9c9c4`, testo zinc-900.

**Palette per grafici (token dedicati `--color-chart-*` esistenti nel CSS):** brand-1 `#ff5229`, brand-2 `#ff8204`, brand-3 `#ffaf01`, brand-4 `#ff6660` (scuro), brand-5 `#ffdd73`; più blue `#55b3fb`/`#0082e6`, green `#6cd2a7`/`#3aa475`, lavender `#c86ee2`/`#b442d7`, pink `#ff56c6`, neutri cream/steel; griglia = bordo primario, assi = bordo secondario, etichette = testo attenuato. **Stile grafici verificato a vista:** card `rounded-xl` (12 px) con bordo 1 px e padding 32 px, titolo 16/24 px, sottotitolo 13 px attenuato ("Lower is better ↓"), barre orizzontali con **arancio per il modello in evidenza e grigio per gli altri**, valori numerici a fianco, pulsante di download (32 px, `rounded-md`), pill di selezione tra metriche.

**Tipografia**
| Uso | Font | Misura |
|---|---|---|
| Titoli, titoli grafici | **ALTMistral** (custom) | h1 72/72 px, tracking −1,44 px, peso 500 (misurato a 800 px di larghezza); h2 44/52 px; h3 24/28; h4 20/24 |
| Corpo, UI | **Inter** variabile (100–900) | 16/24 px |
| Etichette, eyebrow, tag, menu | **Space Mono** 400 | 11–14 px, maiuscolo, tracking ~0,02 em |

**Forme e ritmo:** raggi 2/4/6/8/12/16/24/32 px (md = 6 px per bottoni e input, xl = 12 px per card); unità di spaziatura 4 px; bottoni alti 40 px (nav 48 px), padding 8×12; contenitori fino a 72 rem; transizioni `cubic-bezier(.4,0,.2,1)`; movimento breve e lineare. I breakpoint responsive non sono stati mappati per intero.

### 3.3 Cosa si può replicare liberamente e cosa no
*(Non è consulenza legale.)*
| Elemento | Stato | Proposta |
|---|---|---|
| Token di colore, spaziatura, raggi, griglia, componenti, stile grafici, motion | Sono scelte di design osservabili | Replico **1:1** |
| Inter, Space Mono | Open source (SIL OFL) | Incluse nel progetto (offline) |
| **ALTMistral** | Font proprietario di Mistral; i file sono serviti dal loro sito, non ho trovato una licenza pubblica | **Aggiornato (§3.5):** scaricato per uso privato di test, mai pubblicato. Stack `"ALTMistral", Inter`: Inter Medium resta il ripiego se i file mancano (es. dopo un clone del repo) |
| Logo / wordmark, illustrazioni pixel dei modelli, product shots | Marchi e asset di Mistral; l'uso esplicito è concesso a partner/collaboratori | **Aggiornato (§3.5):** si usa il logo ufficiale Timestral (solo icona/banner, scaricati) con dicitura "non affiliato"; illustrazioni degli altri modelli e product shots non usati; le altre icone sono **disegnate da noi** in stile pixel |
| Nome "Timestral" | Nome di prodotto Mistral | Nome dell'app descrittivo + dicitura "Strumento indipendente, non affiliato a Mistral AI" |

### 3.4 Come garantisco il "1:1" nel tempo
- `brand/tokens.json` con **provenienza** (URL, data di estrazione) generato da `scripts/extract-brand-tokens`, ri-eseguibile se Mistral cambia il sito.
- Controllo di conformità: test che confronta i CSS dell'app con i token + confronto visivo a schermo (browser integrato) con le pagine di riferimento di mistral.ai.

---

### 3.5 Aggiornamento: scelte confermate sul brand (2026-10-27)
- **Logo:** si usa il logo ufficiale Timestral (icona clessidra pixel-art con la "M" nella sabbia). Scaricati in `web/assets/mistral/logos/`: `timestral-icon-model.svg` (5 KB, da mistral.ai; ispezionato: nessuno script o riferimento esterno), `timestral-hourglass.png` (2 KB, dal repo Hugging Face), `timestral-banner.jpg` (220 KB, dal repo GitHub).
- **Font:** scaricati i 6 file **ALTMistral** (woff2: Regular, Italic, Medium, MediumItalic, Semibold, SemiboldItalic, ~76–82 KB ciascuno) in `web/assets/mistral/fonts/`, dallo stesso URL pubblico che il sito serve ai browser. Non esiste una pagina di download né una licenza pubblica trovata: **uso privato per i nostri test, mai pubblicato**. In `web/fonts/` ci sono anche **Inter** variabile (349 KB) e **Space Mono** Regular (35 KB), open source (SIL OFL), committabili.
- **Protezione per GitHub:** `.gitignore` esclude `web/assets/mistral/` (font e loghi), `.venv/`, `.cache/`, dati e risultati. Uno script `scripts/fetch-brand-assets` li riscarica in locale; senza di essi la UI ripiega su Inter + icona neutra, quindi il repository resta pubblicabile.
- **Disclaimer:** nel piè di pagina resta la dicitura "Independent tool, not affiliated with or endorsed by Mistral AI".
- **Tema chiaro: verificato che esiste** sul sito ufficiale: selettore a tre voci (`system` / `light` / `dark`, salvato in `localStorage.theme`). Valori chiari misurati: sfondo `#fbfbf8`, card `#f5f4ef`, terziario `#ebe9e0`, bordo `#e4e3de`, bordo secondario `#c9c9c4`, testo primario oklch(21% .006 285.885), attenuato oklch(55.2% .016 285.938), accento `#ff5229`, chart-brand-4 `#c4001d`, neutri grafico `#bfbcac`/`#e4e2d6`. Quindi nell'app: **scuro di default** con selettore System/Light/Dark come Mistral.
- **Nota sui grafici di Mistral:** sotto i loro grafici dichiarano "Axis starts at 0.44 — bar length is not proportional to the score". Nei nostri grafici le barre partono da zero di default; se un asse è troncato lo diciamo esplicitamente.

---

## 4. Architettura proposta

```
Browser (UI senza build)  <->  FastAPI su 127.0.0.1  <->  Core Python
                                                     ├─ registry modelli (adapter)
                                                     ├─ loader dataset (locale / hub / sintetici)
                                                     ├─ runner backtest (job in background, SSE)
                                                     ├─ metriche + calibrazione
                                                     ├─ profiler (tempo, RAM, VRAM) + sonda hardware
                                                     └─ storage (SQLite + file .npz/.parquet)
Inferenza in processo worker separato (spawn): isola memoria, crash e timeout.
```

### 4.1 Estensibilità ai modelli (R1)
Interfaccia `ModelAdapter` (Protocol) con: `info()` (parametri, layer, contesto, orizzonte per passata, livelli di quantile, licenza, capacità: multivariato / covariate passate / future / categoriche / dispositivi supportati), `load(device, cache_dir)`, `unload()`, `forecast(ForecastRequest) -> ForecastResult`.
- Risultato **normalizzato**: quantili `(B,V,H,Q)` + livelli dichiarati dall'adapter. Le metriche usano solo i livelli disponibili: nessuna assunzione "21 quantili".
- Registrazione tramite **manifest** (`models/*.toml`) + entry point `tsfm_lab.models`: aggiungere un modello = 1 file adapter + 1 manifest.
- Dipendenze dei futuri modelli come *extras* opzionali (es. `[chronos]`) e processo worker dedicato, per evitare conflitti di versione.
- Baseline integrate senza dipendenze extra: naive, seasonal naive, drift. Servono per la skill score e per una lettura onesta dei risultati.

### 4.2 Dataset (R2)
Rappresentazione interna unica (formato lungo: id, timestamp, target, covariate passate/future, categoriche statiche) + metadati: fonte, licenza, hash, frequenza, flag di possibile contaminazione.
1. **Test inclusi (sintetici, seed fissato):** generatori allineati alle 8 caratteristiche strutturali di CaFE (trend, multi-stagionalità, stagionalità variabile, cambio di regime, intermittenza, fattore comune, dipendenza tra serie, risposta alle covariate) + suite di **casi limite**: blocchi mancanti, outlier, level shift, serie costanti, scale estreme, valori negativi, contesto da 32k, e la riproduzione del bug #7 (covariata piatta).
2. **Locali:** CSV/TSV/Parquet/Excel con procedura guidata di mappatura colonne (tempo, id, target, covariate passate/future) e **rapporto di validazione** (buchi, duplicati, % NaN, frequenza dedotta).
3. **Online (Hugging Face Hub, verificati):** `Salesforce/GiftEval` (Apache-2.0), `autogluon/chronos_datasets` e `autogluon/fev_datasets` (licenza "other": mostrata per dataset). Download in cache di progetto, con avviso di dimensione prima di scaricare.
- Badge **"possibile contaminazione"** sui dataset pubblici: i dati reali del post-training di Timestral non sono dichiarati e gli autori di GIFT-Eval riconoscono il problema.

### 4.3 Valutazione e trasparenza (R2)
- **Metriche:** MASE, WQL, CRPS approssimato dai quantili (come l'obiettivo di training dichiarato da Mistral), pinball per quantile, **copertura empirica vs nominale** (50/80/90/98%), larghezza e interval score, bias, **skill score vs baseline**, tutte per orizzonte, per serie e aggregate.
- **Backtest a origine mobile** (finestre configurabili) con seed e ripetibilità.
- **Ogni esecuzione salva:** config completa, revisione (commit SHA) dei pesi, versioni di Python/torch/numpy/CUDA, hash del dataset, previsioni grezze, metriche per serie, tempi. Esportabile in JSON/CSV/PNG e con comando CLI "riproduci".
- **Prestazioni locali:** tempo di caricamento, latenza a freddo/a caldo, throughput vs batch / contesto / orizzonte, **picco RAM/VRAM** (misurato nel worker), thread usati, dispositivo. Un micro-benchmark standard replica lo scenario di Mistral (96 passi da 2016) per confrontare direttamente con 17/113 ms.
- Non pretendo di replicare le classifiche ufficiali: la sandbox mostra **risultati locali**. Un'integrazione opzionale con la libreria `fev` può venire in una fase successiva.

### 4.4 Grafici (ogni dettaglio, stile Mistral)
Fan chart (storico, mediana, bande annidate dei quantili, valori reali, covariate sotto) · errore per orizzonte · diagramma di calibrazione/affidabilità · reale vs previsto · istogramma dei residui · mappa serie × orizzonte · distribuzione delle metriche tra serie (box/violino) e peggiori N serie · scalabilità latenza/memoria · confronto tra modelli (barre arancio per il modello in evidenza, grigio per gli altri, come nell'annuncio) · ablation covariate on/off e `symmetric_averaging` on/off. Controlli per livello di dettaglio, zoom, tooltip numerici, esportazione PNG/CSV.

### 4.5 Pagine
Home · Modelli (specifiche, cache, scarica/rimuovi, licenza) · Dataset · Playground (singola previsione interattiva) · Esperimenti (backtest) · Confronto · Sistema (hardware + benchmark) · Storico run · Info e licenze.

---

## 5. Stack tecnologico (proposta)

| Livello | Scelta | Motivo |
|---|---|---|
| Python | **3.13** (con `uv`) | Coincide con l'ambiente in cui il bug #7 è stato riprodotto su Windows; 3.14 è installato ma con meno garanzie sulle wheel |
| Backend | FastAPI + uvicorn, SQLite (stdlib), numpy, pandas, pyarrow, psutil, torch, safetensors, huggingface_hub | Tutto permissivo; nessun servizio esterno |
| Frontend | **HTML + Tailwind CSS v4 + JavaScript (ES modules)**, senza passaggio di build per il JS. Piccolo helper di rendering **Preact + htm** (JS puro, niente JSX) e **Apache ECharts** inclusi nel progetto (nessun CDN, funziona offline) | Come da tua scelta HTML/CSS/Tailwind/JS. **mistral.ai è costruito con Tailwind v4**: replicando i loro token nel nostro `@theme` (stessi nomi: `surface-brand-secondary`, `text-h3`, `font-mistral`, `rounded-xl`…) il markup dei loro componenti si trasferisce quasi 1:1 |
| Build CSS | **Tailwind standalone CLI** (singolo eseguibile, **senza Node/npm/`node_modules`**), versione fissata e checksum verificato, salvato in `.cache/tools/` | Mantiene il progetto isolato; lo scaricherà lo script di bootstrap |
| Compute | **PyTorch 2.11.0** fissato: wheel **`+cu128`** su Windows/Linux con NVIDIA, wheel CPU su Linux/Windows senza NVIDIA, wheel standard su macOS | 2.11.0 è la versione con cui upstream ha riprodotto i bug su Windows; cu128 supporta le RTX serie 50 (Blackwell). Verificato che esiste per cp313 su Windows e Linux. Una build CUDA esegue anche su CPU |
| Lock | `uv.lock` + SHA del commit di `timestral` | Riproducibilità |

---

## 6. Isolamento e pulizia (R4)

- Ambiente: `uv venv --python 3.13 .venv` (riusa il Python già installato: nessuna installazione di sistema).
- **Tutte le cache dentro il progetto**, impostate da `scripts/env` e **forzate anche dall'app** all'avvio: `HF_HOME`, `HF_HUB_CACHE`, `TORCH_HOME`, `UV_CACHE_DIR`, `PIP_CACHE_DIR`, `XDG_CACHE_HOME`, `MPLCONFIGDIR`, `TMPDIR/TEMP`, `PYTHONPYCACHEPREFIX`, `UV_PYTHON_INSTALL_DIR` (solo se uv dovesse scaricare un Python); `HF_HUB_DISABLE_TELEMETRY=1`.
- Su questa macchina esistono già cache globali di pip e uv: **non verranno toccate né popolate**.
- `scripts/clean` (rimuove `.venv` e `.cache`), `scripts/doctor` (mostra cosa occupa spazio e dove).
- **Verifica di isolamento che farò davvero:** istantanea prima/dopo di `AppData\Local\pip`, `AppData\Local\uv`, `%USERPROFILE%\.cache`, `%TEMP%` e profilo utente, per dimostrare che dopo installazione ed esecuzione non compare nulla fuori dal progetto.
- Limiti onesti: il `localStorage` del browser per `localhost:porta` (preferenze UI) resta nel profilo del browser; è marginale e cancellabile.
- Spazio previsto: pesi Timestral ~1,46 GB + PyTorch (wheel misurate sul server PyTorch: **`2.11.0+cu128` Windows = 2,56 GB**, CPU = 0,11 GB; decompresse occupano di più) + dataset scaricati. Sulla tua macchina (285 GB liberi) non è un problema.

### 6.1 Supporto CPU e GPU (CUDA) fin dall'inizio
- **Un solo ambiente per entrambi:** la build CUDA di PyTorch esegue anche su CPU. Il dispositivo si sceglie **a runtime**, senza reinstallare.
- **Selettore nell'interfaccia** (Playground, Esperimenti, Sistema): `CPU` sempre disponibile; `CUDA:0 — <nome GPU> (<VRAM>)` se rilevata; l'utente può scegliere la CPU **anche con la GPU presente**. Dispositivi non disponibili o non supportati (es. MPS) appaiono disattivati con la motivazione.
- **Cambio dispositivo = scarico e ricarico del modello nel processo worker**, con tempo di caricamento misurato e mostrato.
- **Confronto CPU vs GPU come funzione di prima classe:** stessa configurazione eseguita su entrambi, con latenza a freddo/a caldo, throughput per dimensione di batch, picco RAM e VRAM, e **differenza numerica tra le previsioni** dei due dispositivi (attesa dell'ordine del rumore float32).
- **Misure corrette su GPU:** warm-up, sincronizzazione (`torch.cuda.synchronize`) prima di fermare il cronometro, picco VRAM con `torch.cuda.max_memory_allocated`, ripetizioni con mediana e dispersione.
- **Bootstrap:** rileva NVIDIA (`nvidia-smi`) e sceglie la wheel; `--cpu` per forzare quella leggera.

---

## 7. Piattaforme (R5)

| Piattaforma | Stato atteso | Nota |
|---|---|---|
| Windows (questa macchina) | CPU e CUDA | Verificabile qui |
| Linux | CPU e CUDA | **Non verificabile qui**; serve CI o un tuo test |
| macOS | CPU | MPS disattivato di default per il bug #6; **non verificabile qui** |

Macchina di sviluppo: Windows 11, Python 3.13.15 / 3.14, uv 0.12.18, Node 22, git 2.56, **RTX 5060 Ti 16 GB** (Blackwell: serve una build CUDA ≥ 12.8), Ryzen 5 3600 (6C/12T), 48 GB RAM, 285 GB liberi. All'inizio la cartella di progetto era vuota e non è ancora un repository git (finora contiene `RESOCONTO.md`, `.gitignore` e `web/` con font e loghi).

---

## 8. Rischi

| Rischio | Gestione |
|---|---|
| Pacchetto giovane (v1.0.0), issue aperte, nessuna PR accettata | SHA bloccato; layer adapter che isola le modifiche; ricontrollo delle issue prima di ogni aggiornamento |
| Risultati fuorvianti (contaminazione, margini minimi tra modelli) | Badge, baseline sempre presenti, calibrazione in evidenza, nessuna pretesa di replicare le classifiche |
| Licenza pesi | Avviso in Modelli e Info; verifica da parte tua |
| Asset di Mistral (logo, font) in un repo pubblico | Usati solo in locale; esclusi da `.gitignore`; la UI funziona anche senza (ripiego su Inter); nessuna pubblicazione dei file |
| Il sito Mistral cambia design | Token con provenienza + script di ri-estrazione |
| Cross-platform non testato | CI su Windows/Linux/macOS dopo `git init` + GitHub (decisione tua) |

---

## 9. Decisioni

### Confermate (2026-10-27)
| # | Decisione |
|---|---|
| 1 | **Logo ufficiale Timestral** nell'app (scaricato, §3.5), con dicitura "non affiliato" nel piè di pagina |
| 2 | **Font ufficiale ALTMistral** scaricato per uso privato di test; mai pubblicato (esclusione in `.gitignore`) |
| 3 | Frontend **HTML + Tailwind + JS**; stack dettagliato in §5 |
| 4 | **Supporto CUDA fin dall'inizio** con selettore CPU/GPU nell'interfaccia, anche con GPU presente (§6.1) |
| 5 | **Lingua dell'interfaccia: solo inglese** |
| 6 | **Tema scuro di default**, con selettore chiaro perché il tema chiaro esiste sul sito Mistral (verificato, §3.5) |
| 7 | Python 3.13 e `git init` (raccomandazioni accettate implicitamente: da confermare insieme al nome del repo) |

### Ancora aperte
- **Nome del progetto e del repository GitHub.** Determina il nome del pacchetto Python, le cartelle e i prefissi delle variabili d'ambiente. Proposta: `tsfm-lab` (pacchetto `tsfm_lab`).
- **Via alla Fase 0:** scaricherà PyTorch `2.11.0+cu128` (2,56 GB), il pacchetto `timestral` da git, Tailwind CLI e i pesi Timestral-5M (21 MB) per il test.

---

## 10. Piano di lavoro

| Fase | Contenuto | Criterio di completamento |
|---|---|---|
| 0 | Bootstrap, `.venv`, cache locali, torch cu128/CPU, script `env/run/clean/doctor/fetch-brand-assets`; test minimo di Timestral-5M su **CPU e CUDA** | Due comandi su Windows pulito; la GPU viene rilevata; **nessuna traccia fuori dalla cartella (verificata con istantanee)** |
| 1 | Core: adapter, Timestral, baseline, dataset locali e sintetici, metriche, backtest, storage, CLI, test | Previsione col 5M con forme `(B,V,H,21)` come da README; test unitari su metriche; test con modello finto senza scaricare nulla |
| 2 | API + shell UI con token Mistral + pagine Modelli, Playground, Sistema | Pagine fedeli ai token; micro-benchmark confrontabile con 17/113 ms |
| 3 | Esperimenti, tutti i grafici, storico ed esportazioni | Ogni run riproducibile e completamente ispezionabile |
| 4 | Dataset online, badge contaminazione, suite casi limite, preset covariate | Riproduzione del bug #7 e della sua mitigazione dentro l'app |
| 5 | Rifinitura: controllo di conformità del brand, CI, documentazione, secondo adapter di prova (es. Chronos-2) | Aggiungere un modello richiede 1 adapter + 1 manifest, dimostrato da un test |

---

## 11. Fonti
- Annuncio Mistral: https://mistral.ai/news/timestral/ · Brand: https://mistral.ai/brand
- Repo: https://github.com/mistralai/timestral/ (issue 4–7) · Model card: https://huggingface.co/mistralai/Timestral-209M
- Licenza pesi (testo grezzo): https://huggingface.co/mistralai/Timestral-209M/raw/main/LICENSE
- CSS e stili calcolati di mistral.ai (misura diretta del 2026-10-27)
- Valutazione indipendente: https://philippdubach.com/posts/forecasting-financial-markets-with-timestral/ · CaFE: https://arxiv.org/html/2610.26251
- Dataset: https://huggingface.co/datasets/Salesforce/GiftEval · https://huggingface.co/datasets/autogluon/chronos_datasets · https://huggingface.co/datasets/autogluon/fev_datasets

---

## 12. Avanzamento

### Fase 0 — completata (2026-10-27)
**Ambiente (tutto in `.venv/` e `.cache/`, bloccato in `uv.lock`):** Python 3.13.15, PyTorch **2.11.0+cu128**, `timestral` 1.0.0 al commit `a534bdb` (HEAD upstream al momento), Tailwind CLI 4.3.3 verificato contro l'hash SHA-256 fissato nello script. Spazio reale su disco: **4,5 GB** (`.venv` e cache uv condividono i file con hardlink).

**Isolamento — verificato con `scripts/leakcheck.py` (istantanea prima e dopo l'installazione):** cache pip, uv (due posizioni), cache JIT CUDA di NVIDIA, `~/.cache`, `~/.triton`, `~/.nv`, profilo utente e file temporanei: **tutti invariati**. La cache JIT di CUDA (59 MB) è finita in `.cache/cuda`. Un problema evitato: pytest avrebbe scritto nella cartella temporanea di sistema; ora usa `.cache/pytest`.

**Smoke test Timestral-5M** (RTX 5060 Ti 16 GB, Ryzen 5 3600 con 6 thread, 20 ripetizioni dopo 2 di riscaldamento, GPU sincronizzata):

| | CPU | CUDA |
|---|---|---|
| Caricamento modello | 2,13 s | 9,52 s (include l'inizializzazione di driver e CUDA) |
| 1 serie, contesto 2016, orizzonte 96 (mediana) | 31,2 ms | 47,8 ms |
| 64 serie, stesso scenario (mediana) | 482,8 ms (132,6 serie/s) | 49,1 ms (**1302 serie/s**) |
| Picco VRAM (1 serie / 64 serie) | – | 29,9 MB / 85,1 MB |

- **Coerenza tra dispositivi:** differenza massima tra previsioni CPU e CUDA `3,05e-5` (`4,9e-7` dell'ampiezza della serie): rumore float32. Forme `(B,1,96,21)`, valori finiti e quantili non incrociati su entrambi i dispositivi.
- **Lettura (ipotesi, non dimostrata):** per una singola serie piccola la GPU è *più lenta* della CPU; il tempo CUDA resta ~48 ms anche a 64 serie, il che fa pensare a una latenza dominata da overhead (lancio dei kernel, decodifica a blocchi) e non dal calcolo. Con i batch la GPU è circa 10 volte più veloce in throughput. È una misura singola sul solo modello 5M: va ripetuta nell'app su tutte e quattro le taglie.
- **Confronto con Mistral:** 17 ms dichiarati su una CPU server a 8 thread contro 31 ms qui su un Ryzen 5 3600 a 6 thread. Stesso ordine di grandezza, ma hardware diverso: non è un confronto 1:1.
- Test unitari: 4/4 superati.

**Scoperte utili per le fasi successive:** la prima esecuzione su CUDA ricompila kernel per la Blackwell (cache in `.cache/cuda`), quindi il tempo di caricamento a freddo va mostrato separato da quello a caldo; il bootstrap risolve e blocca 87 pacchetti in ~2 s.

### Fase 1 — completata (2026-10-27)
**Costruito** (`src/tsfm_lab/`): contratto `ModelAdapter` indipendente dal modello (capacità, quantili dichiarati dall'adapter, dispositivi) e registro con entry point `tsfm_lab.models`; adapter **Timestral** per le 4 taglie (manifest TOML, caricamento su CPU o CUDA scelto a runtime, suddivisione automatica in blocchi, controlli sulle covariate, specifiche lette dal `config.json` reale); baseline **naive / seasonal-naive / drift**; 16 scenari sintetici (le 8 caratteristiche strutturali + 8 casi limite); caricamento di CSV/TSV/Parquet/Excel con mappatura colonne e rapporto di validazione; metriche (MASE, SQL, WQL, copertura 50/80/90/98, interval score, bias, calibrazione, skill score) con definizioni documentate; backtest a origine mobile con finestre identiche per tutti i modelli; salvataggio dei run (SQLite + array compressi + JSON rigoroso); sonda hardware/software; CLI `tsfm-lab` (`models`, `devices`, `datasets`, `inspect`, `run`, `runs`, `show`).

**Verifica:** **52 test superati** (CPU e GPU), compreso un test che dimostra la promessa di estensibilità (un modello finto si aggiunge con un solo adapter, senza toccare altro) e uno che verifica che la suddivisione in blocchi non cambia le previsioni. `leakcheck` ripetuto dopo tutto il lavoro: nulla fuori dal progetto.

**Difetti trovati dai test e corretti:** il generatore `negative_values` produceva valori in parte positivi; la mediana delle baseline non coincideva con la previsione puntuale (residui non centrati). Sono emersi anche due problemi di misura: il tempo su CUDA includeva l'avvio a freddo (16,7 s contro ~50 ms a caldo), quindi il backtest ora fa un passaggio di riscaldamento non misurato e riporta il tempo a freddo a parte.

**Primi risultati (Timestral-5M, dati sintetici: indicano che il motore funziona, non che il modello è "migliore"):**

| Scenario | Modello | MASE | Copertura 80% / 90% | Skill vs seasonal-naive |
|---|---|---|---|---|
| Doppia stagionalità (CPU) | Timestral-5M | 0,256 | 0,77 / 0,88 | +78,9% |
| | seasonal-naive | 1,263 | 0,75 / 0,85 | – |
| Dipendenza tra variate (CUDA) | Timestral-5M | 0,920 | 0,81 / 0,91 | +20,1% |
| | seasonal-naive | 1,165 | 0,77 / 0,87 | – |

Il modello, pre-addestrato soprattutto su dati sintetici, può trovare "facili" questi scenari: i risultati su dati reali vanno letti a parte.

**Bug upstream #7:** lo scenario `flat_price_covariate` è pronto e la mitigazione (`mask_flat_covariates`) funziona (q99 massimo 23,3 → 7,0, contro 6,6 senza covariate). Con il **28M non si riproduce l'esplosione** (coerente con quanto riportato upstream, dove i casi estremi erano sul 209M). Riprodurre sul 209M richiede il download dei suoi pesi (837 MB): in attesa del tuo ok.

### Fase 2 — completata (2026-10-27)
**Pesi:** scaricati tutti e 4 i checkpoint Timestral nella cache del progetto (5M 20 MB, 28M 106 MB, 123M 468 MB, 209M 798 MB), come da tuo ok.

**Costruito:** API FastAPI locale (solo 127.0.0.1, rifiuta Host e Origin diversi da localhost) e interfaccia web **HTML + Tailwind v4 + JavaScript** (Preact/htm e ECharts inclusi, senza CDN né Node). Comando: `python -m tsfm_lab serve`. Pagine: **Overview**, **Models** (specifiche reali dai checkpoint, download/rimozione pesi, avvisi su licenza e problemi noti, benchmark del vendor *etichettati come non misurati*), **Datasets** (16 scenari, anteprima, caricamento di file con mappatura colonne e rapporto di validazione), **Playground** (previsione singola con ventaglio di intervalli 50/80/90/98%, covariate, confronto con baseline), **Experiments** (backtest in background con avanzamento e annullamento), **Runs** (riepilogo, accuratezza, errore per passo, calibrazione, copertura, distribuzione, dispersione, residui, navigatore delle singole previsioni, esportazione CSV/JSON, blocco di riproducibilità), **System** (hardware, dispositivi, disco, **benchmark CPU contro GPU** con avvio a freddo separato), **About** (licenze e glossario delle metriche).

**Design Mistral:** token estratti dal loro CSS con gli stessi nomi (superfici, bordi, testo, azioni, palette dei grafici), font ALTMistral/Inter/Space Mono, navbar a celle con linee sottili, card da 12 px, etichette mono maiuscole, icone pixel disegnate da noi, barre dei grafici da zero (arancio per il modello principale). **Tema scuro di default** con selettore Sistema/Chiaro/Scuro come sul loro sito (il chiaro è stato verificato a schermo).

**Mobile:** viewport corretto, navbar che diventa menu a comparsa sotto 1280 px, griglie a colonna singola, tabelle con scorrimento interno, grafici più compatti, bersagli touch da 40 px, salto automatico al risultato dopo una previsione. Verificato a 375 px: **nessuna pagina ha scorrimento orizzontale** (controllo automatico su tutte le 9 pagine).

**Selettore CPU/GPU:** nella barra superiore (e nei form), sempre disponibile anche con la GPU presente; i dispositivi non supportati compaiono disattivati con la motivazione.

**Verifica:** 64 test (CPU e GPU), compresi quelli dell'API (sicurezza locale, flusso esperimento→run→analisi→CSV, upload e mappatura, benchmark, annullamento) e uno che verifica che l'interfaccia e le librerie vengano servite. Interfaccia provata nel browser integrato (desktop scuro e chiaro, mobile): nessun errore in console. `leakcheck` ripetuto: nulla fuori dal progetto.

**Difetti trovati provando davvero l'app e corretti:** `/api/system` scorreva 40.000 file a ogni apertura (ora il disco è un endpoint a parte); dimensioni dei pesi "0 B" su Windows (gli snapshot HF non usano symlink); navbar che sforava a 1024 px; colonne di griglia implicite che si allargavano al grafico (regola base per tutte le griglie); navigazione tra Dataset e Playground che non cambiava dataset; fan chart che mostrava tutta la storia invece della previsione.

**Bug upstream #7, riprodotto in app e con i numeri (209M, CUDA, 16 serie, scenario `flat_price_covariate`):**

| Configurazione | q99 massimo mediano sulle serie a rischio | Serie con q99 > 50 (su 32) | SQL |
|---|---|---|---|
| Predefinita (symmetric on) | **7.265** | 17 | 13,25 |
| `mask_flat_covariates` | 6,5 | 0 | 0,483 |
| `symmetric_averaging` off | 16,7 (worst 4.199) | 7 | 0,627 |
| Senza covariate | 6,6 | 0 | 0,482 |

La domanda reale è Poisson(2). Sul 123M: 12,1 (worst 95,7) predefinito, 6,2 con la maschera. Il 28M non esplode. Disattivare solo la simmetria **non basta**. Nell'app si vede aprendo il run: avvisi espliciti, SQL 7,7 contro 0,77 della baseline, pur con MASE buono (+27,9%) perché la mediana resta sensata.

**Benchmark CPU contro GPU (Timestral-5M, Ryzen 5 3600 / RTX 5060 Ti):** setup di Mistral 29,6 ms su CPU contro 50,9 ms su GPU a caldo; avvio a freddo 30 ms (CPU) e 129 ms (GPU); a batch 128 la GPU fa **2.232 serie/s contro 129** (17 volte). Previsioni CPU e GPU identiche entro `3e-5`. Misure singole sul 5M.

**Deviazioni dal piano e limiti noti (onestà):**
- ~~L'inferenza gira nello stesso processo del server~~ **Risolto dopo la Fase 2:** ora gira in un processo worker separato (vedi sotto).
- **Dataset online** (Hugging Face) e badge di contaminazione per i dataset pubblici: Fase 4. Per ora i file caricati sono marcati "unknown".
- L'upload di file è coperto dai test API, ma non l'ho provato cliccando nel browser (gli strumenti non consentono di scegliere un file).
- Linux e macOS non testati; su Mac la GPU (MPS) resta disattivata per il bug upstream #6.
- ALTMistral e logo funzionano solo in locale (file ignorati da git); senza di essi l'interfaccia ripiega su Inter.

### Prossimi passi
Fase 3: pagina di confronto tra run, ulteriori viste e rifiniture di visualizzazione. Fase 4: dataset online, badge di contaminazione, preset delle covariate. Fase 5: processo worker per l'inferenza, controllo di conformità del brand, CI su Windows/Linux/macOS, secondo adapter di prova (es. Chronos-2).

### Isolamento dell'inferenza e accesso da Tailscale (2026-10-27)
- **Worker separato:** previsioni, esperimenti, benchmark e download girano in un processo dedicato (avviato al primo uso con `spawn`, nessuno stato CUDA ereditato). Un crash nativo di PyTorch/CUDA ferma solo il worker: il server risponde con un errore chiaro e il worker riparte alla richiesta successiva. Avanzamento e annullamento passano per coda ed evento. Verificato con un test che simula un crash (`exit code 13`) e dal vivo su GPU: prima chiamata 6,2 s (avvio worker + caricamento), poi 40 ms; annullamento di un benchmark lungo dopo ~11 s (attende la fine della cella in corso).
- **Tailscale:** `python -m tsfm_lab serve --tailscale` ascolta solo su loopback e sull'IP Tailscale del PC (non sulla LAN), accetta solo quei nomi host e rifiuta gli altri. Non c'è autenticazione: chiunque sia nella tailnet può usare l'app (eseguire job, caricare file, cancellare pesi). Non usare Funnel.

### Fase 3 — confronto tra run (2026-10-27)
- **Pagina Confronto** (`#/compare?runs=A,B,…`, fino a 6 run, selezionabili dall'elenco Runs): tabella **«cosa è cambiato»** che evidenzia solo i campi diversi (modello, dispositivo, dataset, orizzonte, finestre, opzioni…), avvisi se i run non sono confrontabili (dataset, orizzonte o finestre diversi), barre per metrica, tabella con calcolo/avvio a freddo/VRAM e **confronto appaiato previsione per previsione** (dispersione con diagonale, quota di finestre vinte e guadagno tipico come media geometrica): una media può nascondere un modello che perde nella maggior parte delle finestre.
- **Rilancia con le stesse impostazioni:** ogni run salva la richiesta originale; il dettaglio ha il pulsante per rieseguirla.
- Provata sul caso CPU contro GPU dello stesso esperimento (28M, `regime_switching`): 36 finestre identiche, appaiate. Test API aggiunto (66 in totale).

### Showcase (2026-10-27)
**Obiettivo:** mostrare le potenzialità dei modelli con grafici comprensibili anche a non tecnici, come sul sito di Timestral. Ora è la **pagina iniziale**; la parte tecnica resta dietro (Playground, Esperimenti, Runs, Confronto).

- **Tre scenari reali** da `autogluon/fev_datasets` (revisione fissata `d3e7e720f4`, in totale ~10 MB scaricati su richiesta): **vendite di negozi Rossmann** (apertura, promozioni, vacanze scolastiche e festività, 40 negozi), **domanda di elettricità** di 6 paesi (solare, eolico, temperatura, irraggiamento) e **produzione di un impianto solare** (luce, nuvole, temperatura, umidità, vento).
- **Cosa si vede:** storia in bianco, previsione arancio con fasce al 50% e 80%, un interruttore per ogni fattore noto in anticipo con il suo andamento, scelta di serie e modello, cursore per la data di inizio previsione, «mostra cosa è successo davvero». Le previsioni si ricalcolano a ogni interruttore (frazioni di secondo su GPU).
- **Il numero «% di errore in meno» è misurato, non scelto:** quality loss ponderata su tutte le previsioni delle ultime finestre di ogni serie (400 per le vendite, 180 elettricità, 40 solare), con e senza fattori, e per ogni fattore da solo; salvata su disco (`runs/showcase/`), quindi la seconda volta è istantanea.
- **Risultati misurati** (perdita quantilica, meno è meglio): vendite **−29% con il 28M, −41% con il 209M** (apertura, promozioni e festività contano, le vacanze scolastiche no); solare **−45% / −48%** (quasi tutto dalla luce); domanda elettrica **solo −2% / −4%**: con un ritmo giornaliero e settimanale molto regolare i fattori aiutano poco, e l'app lo dice. In elettricità la fascia all'80% conteneva la realtà solo il 63% delle volte: segnalata come troppo stretta.
- **Errore evitato:** la colonna `target` di `entsoe` sembrava un prezzo ma è la **domanda** (mai negativa, −14/−19% nel weekend): lo scenario è etichettato così, senza unità inventata. Il dataset di Tetuan mostrato da Mistral non è disponibile nelle fonti: sostituito dal solare con meteo.
- **Avvertenze mostrate all'utente:** per elettricità e solare i fattori sono valori *misurati*, non previsioni, quindi il beneficio è un limite superiore; i dataset sono pubblici e potrebbero essere nei dati di addestramento (non dichiarati); licenza `other` dichiarata sul repository, termini delle fonti originali da verificare.
- **Provato** nel browser integrato (desktop e 375 px, nessuno scorrimento orizzontale, nessun errore bloccante); 68 test.

### Lavori in background, ordine dei modelli e header (2026-10-27)
- **Pannello «Attività» globale:** ogni benchmark, esperimento, download o misura in corso compare in basso a destra **in ogni pagina**, con avanzamento reale, Stop e «Open page»; gli esiti recenti restano 3 minuti con il link al risultato. Lo stato vive sul server: cambiando pagina o **ricaricando** si ritrova tutto, e le pagine System, Experiments e Models ripropongono avanzamento e Stop. I redirect automatici ora scattano solo se la pagina che ha avviato il lavoro è ancora aperta.
- **Stop vero:** se un calcolo lungo non guarda il segnale di annullamento, dopo 4 s il processo worker viene terminato e riparte alla richiesta successiva (provato dal vivo: benchmark fermato in 7,7 s). Il benchmark ora indica *prima* quale cella sta misurando.
- **Modelli dal più grande al più piccolo** ovunque (209M, 123M, 28M, 5M): menu, schede, benchmark, CLI.
- **Header:** link più compatti e selettore del dispositivo più stretto: a 1280 px il selettore del tema sta dentro il bordo.

### Showcase: tenda trascinabile e timeline con valori (2026-10-27)
- **Tenda come sul sito ufficiale:** una linea verticale con il quadratino bianco e l'etichetta «Forecast» sul grafico. A sinistra la realtà (bianca), a destra la previsione (arancio) con la realtà in **grigio chiaro** sotto. Si **trascina** (mouse e touch) per cambiare da dove parte la previsione, con la data che segue il quadratino; da tastiera con le frecce (Home/End per i limiti). Sostituisce il cursore separato.
- **Timeline con i valori:** etichette dell'asse del tempo più leggibili e **cursore sincronizzato** su grafico e righe dei fattori; sotto compaiono data (con l'ora per i dati orari), valore reale, previsione con l'intervallo probabile e il valore di ogni fattore in quell'istante (Yes/No per quelli binari).
- «Mostra cosa è successo davvero» è ora attivo di default.

### Showcase: finestra fissa e tenda libera (2026-10-27)
- **Correzione di progetto:** prima si muoveva la *finestra dei dati* e la tenda restava ferma; sul sito ufficiale è il contrario. Ora il grafico mostra una **finestra fissa** (gli ultimi dati) e la tenda si muove liberamente sopra: a sinistra ciò che il modello vede, a destra ciò che prevede, **fino al bordo della finestra**. Più la tenda va a sinistra, più lunga è la previsione (da 7 giorni a 10 settimane sulle vendite). Nelle righe dei fattori la parte dopo la tenda è più tenue.
- **Grafico largo quanto la card**, con righe dei fattori sotto con la stessa scala del tempo; la tenda è posizionata in percentuale dentro l'area del grafico, quindi non può uscire dai bordi (verificato agli estremi con Home/End e su mobile).
- Il confine `k` e l'orizzonte arrivano dall'API (`window`, `k_default`, `k_min`, `k_max`); la misura «% di errore in meno» resta calcolata sull'orizzonte predefinito.

### Showcase: tre nuovi scenari e correzione del bug #7 (2026-10-27)
- **Candidati ispezionati prima di aggiungerli** (schema, lunghezze, valori mancanti, variabilità di ogni fattore) da `autogluon/fev_datasets` (revisione fissata). **Esclusi:** `uk_covid_nation` (nessun fattore noto in anticipo, solo storici) e `m5` giornaliero (84 MB, serie intermittenti: usata la versione settimanale da 9,8 MB).
- **Nuovi scenari:** **Groceries** (`rohlik_sales`, 30,5 MB: vendite giornaliere di 40 prodotti di un negozio online; fattori prezzo, sconto, festività, vacanze invernali), **Store traffic** (`favorita_transactions`, 0,2 MB: acquisti giornalieri di 51 supermercati in Ecuador; fattori festività e prezzo del petrolio) e **US retail** (`m5` settimanale, 9,8 MB: 30 articoli dei negozi della California; prezzo, giorni di pagamento degli aiuti alimentari SNAP, eventi nazionali, religiosi, culturali e sportivi). Sei schede in totale.
- **Serie scelte con una regola che non guarda i fattori** (i più venduti con storia completa), così il risultato non è «ritagliato» per mostrare un guadagno. Fattori non noti in anticipo esclusi di proposito (disponibilità a magazzino, ordini totali); fattori con valori mancanti per la maggior parte delle serie (negozi chiusi) esclusi.
- **Risultati misurati** (GPU, perdita quantilica pesata, «% di errore in meno con tutti i fattori», meno errore è meglio):

  | Scenario | 5M | 28M | 123M | 209M | Fattore che conta |
  |---|---|---|---|---|---|
  | Store traffic | +0,4% | **−2,5%** | +3,4% | +7,2% | festività (da +2% a +8%); prezzo del petrolio sempre negativo (−0,4% … −6%) |
  | US retail | +8,9% | +15,4% | +13,3% | +17,0% | il prezzo da solo (+14% … +20%) |
  | Groceries | +6,7% | +8,3% | +10,9% | +11,4% | lo sconto (+8% … +9%) |

  Le tre schede originali (209M) non cambiano: elettricità +4%, vendite +41%, solare +48%. **Lettura onesta:** il guadagno dipende dal modello (su Store traffic i modelli piccoli non sfruttano le festività e il totale può peggiorare di poco: l'interfaccia ora lo dice a parole, «higher error», e c'è un'avvertenza); il prezzo del petrolio è inutile con ogni modello, ed è lì apposta; con il prezzo da solo US retail migliora più che con tutti i fattori insieme.
- **Bug upstream #7 incontrato dal vivo:** su Groceries il fattore «sconto» *peggiorava* l'errore del **61%**, perché per alcuni prodotti lo sconto era assente in tutta la storia e compariva nell'orizzonte (un solo prodotto, il panino «Roll 8» nei suoi tre magazzini, causava quasi tutto il danno). Con `mask_flat_covariates` il danno sparisce (−61% diventa +9%). **Ora lo Showcase lo attiva di default** (`model_options`), in previsione e misura, e la cache delle misure ne tiene conto. Verificato che gli altri tre scenari non cambiano di un decimale.
- **Dati settimanali:** la durata della previsione si legge «12 weeks» (prima avrebbe detto giorni) e i tasti rapidi muovono la tenda di 4 settimane.
- **Errore corretto:** cambiando scheda restavano visibili, per un attimo, i numeri della scheda precedente (misura vecchia attribuita al nuovo scenario); ora vengono azzerati.
- **Test:** controllo di coerenza di ogni scenario (serie predefinita presente, limiti della tenda, chiavi dei fattori uniche); 63 test non-modello passano.

### CI manuale su Windows, Linux e macOS (2026-10-27)
- **Solo a richiesta:** `.github/workflows/ci.yml` parte esclusivamente con «Run workflow» dalla scheda Actions di GitHub (o `gh workflow run ci.yml -f systems=all|linux|windows|macos`), mai a ogni push, per non consumare i minuti del piano. Il lavoro di ogni sistema è in `check.yml` (workflow riutilizzabile).
- **Cosa fa:** la vera procedura di installazione (`scripts/bootstrap.py`: `uv sync --locked`, Tailwind CLI con checksum fissato), il report `doctor.py`, 64 test senza pesi e i test con il vero checkpoint Timestral-5M (21 MB) sulla CPU. I runner di GitHub non hanno GPU: la GPU resta verificata solo sulla mia macchina.
- **Prima esecuzione:** la prima prova su Linux ha trovato un test che dipendeva dai pesi già scaricati (non un difetto del sistema operativo): corretto. Poi **tutti e tre i sistemi verdi**: Linux 53 s, Windows 88 s, macOS 41 s; 61 test + 6 test con il modello 5M passati su ciascuno (2 + 1 saltati). Installazione pulita, Tailwind verificato, `timestral` alla revisione fissata.
- **Costo:** per i repository privati i minuti si moltiplicano (Linux ×1, Windows ×2, macOS ×10): una esecuzione completa vale circa 15 minuti dei 2000 mensili del piano gratuito.

### Apple GPU (MPS): cosa ho trovato (2026-10-27)
- Il pacchetto ufficiale non funziona su MPS perché calcola statistiche in float64, che MPS non ha (issue #6). Esiste un fork di una riga di commit (`paschembri/timestral@c49d40f`, ramo `mps-support`) costruito sullo **stesso commit** che usiamo, che manda quei calcoli in CPU e riporta i risultati sulla GPU. Modifica 2 file di codice (`forecaster.py`, `scaler.py`) più i test; nessuna nuova dipendenza. Chi lo ha provato su un MacBook Pro M3 Max riferisce risultati uguali alla CPU entro il rumore float32 e **2-2,5 volte più veloce** (di più con molte covariate). Su M5 nessun dato trovato.
- **Integrato su richiesta dell'utente (opzione 1), solo su macOS:** `pyproject.toml` fissa `timestral` così: Windows e Linux restano sul repository ufficiale (`a534bdb`), macOS usa `paschembri/timestral@c49d40f`. **Verifica di sicurezza prima di fidarsi:** è l'unico commit sopra il commit ufficiale che già usavamo (genitore = `a534bdb`, e `main` ufficiale non è andato avanti), tocca 3 file (`forecaster.py` e `scaler.py` di codice, un file di test), nessuna nuova dipendenza, e nelle righe aggiunte non compaiono rete, processi, variabili d'ambiente, apertura di file, pickle o import dinamici (cercati uno per uno): sposta i calcoli float64 sulla CPU e riporta il risultato sul dispositivo. Il commit non è firmato (come la maggior parte); l'account è del 2014 con 44 repository. Il blocco per commit rende impossibile cambiare il codice dopo la verifica. Quando upstream integrerà la correzione basta togliere la seconda riga.
- **Verificato in CI su macOS (Apple Silicon, CPU):** installa il fork, `mps_usable()` risulta vero lì e falso su Windows e Linux (la CI lo controlla a ogni esecuzione). **Non verificato:** l'uso vero della GPU Apple, perché i runner virtuali non ne hanno una: da provare sul MacBook con `python scripts/check_device_parity.py --device mps` (confronta CPU e GPU e misura la velocità).

### Selezione CPU/GPU e ripiego sulla CPU su ogni sistema (2026-10-27)
- **Selettore sempre disponibile** su Windows, Linux e macOS: CPU sempre; GPU NVIDIA (`cuda:N`) se presente; GPU Apple (`mps`) se il pacchetto installato la supporta. Il riconoscimento è per funzionalità, non per piattaforma.
- **Ripiego senza errori:** se la GPU scelta non esiste, non si carica (driver, memoria, operazione non supportata) o fallisce durante la previsione, il motore continua sulla CPU e lo scrive nell'avviso di quel risultato («… the forecast ran on the CPU instead» / «… continued on the CPU»), nel Playground, negli esperimenti e nello Showcase (riquadro «Running on the CPU»). Un vero errore sulla CPU (dati degenerati, bug) non viene nascosto. Provato end-to-end: una richiesta `mps` su Windows risponde con previsione dalla CPU e avviso.
- **Eccezione voluta:** i benchmark CPU contro GPU sono severi: se la GPU non c'è, falliscono invece di etichettare numeri della CPU come numeri della GPU.
- **Robustezza:** un driver CUDA rotto non nasconde più la CPU nell'elenco dei dispositivi; su macOS `PYTORCH_ENABLE_MPS_FALLBACK=1` (rispettando una scelta diversa dell'utente) manda sulla CPU le singole operazioni che la GPU Apple non ha; su MPS le chiamate sono spezzate in blocchi da 64 righe (la memoria è condivisa con il sistema).
- **La CI ha già trovato un difetto mio specifico di macOS** (la variabile aggiunta rompeva il contratto di isolamento, test rosso solo lì): corretto e verde. 77 test locali, CI verde su Linux, Windows e macOS.

### Misura sul MacBook Air M5 (2026-10-27)
Installazione dal repository (`python3.13 scripts/bootstrap.py`, Python 3.13 di Homebrew; la 3.14 non è supportata) e `scripts/check_device_parity.py --device mps` eseguito dall'utente. Il bootstrap ha installato il fork `paschembri/timestral@c49d40f` e ha scaricato Tailwind con checksum verificato.

| Modello | Caso | Differenza CPU/GPU | CPU (regime) | GPU Apple (regime) | Velocità GPU |
|---|---|---|---|---|---|
| 5M, 32 serie | tre casi | ≤ 2,8·10⁻⁵ | 0,03–0,11 s | 1–1,8 s (una sola chiamata, con la compilazione) | non misurato bene |
| 28M, 64 serie | senza covariate / con covariata / 1200 passi | 4,6·10⁻⁷ / 9,2·10⁻⁷ / 4,7·10⁻⁵ | 0,137 / 0,522 / 0,080 s | 0,107 / 0,399 / 0,082 s | 1,28× / 1,31× / 0,98× |
| 209M, 128 serie | senza covariate / con covariata / 1200 passi | 3,1·10⁻⁷ / 6,9·10⁻⁷ / 5,4·10⁻⁶ | 1,410 / 5,670 / 0,355 s | 1,060 / 4,271 / 0,269 s | 1,33× / 1,33× / 1,32× |

- **Correttezza confermata:** la GPU Apple dà le stesse previsioni della CPU entro il rumore float32, anche con covariate e con orizzonti lunghi decodificati a blocchi; il fork funziona su M5.
- **Velocità:** modesta, circa **+30%** sul 209M (a regime, dopo la compilazione dei kernel). Sul 28M da +30% a pari, sul 5M conviene la CPU. Il primo utilizzo di ogni forma di dati costa circa mezzo secondo in più sulla GPU (compilazione Metal). Su un MacBook Air senza ventola il regime prolungato non è stato misurato.
- **Decisione:** il dispositivo predefinito resta la CPU su ogni sistema; la GPU Apple è un'opzione nel selettore, utile soprattutto con i modelli grandi (per confronto: sulla RTX 5060 Ti lo stesso 28M va da 1,7× a 22× la CPU secondo il caso).

### Catalogo dei dataset online (2026-10-27)
- **Lista finita e verificata, non un mercato:** 100 task di **fev-bench** (`autogluon/fev` a un commit fissato, file `benchmarks/fev_bench/tasks.yaml`) su 96 file di `autogluon/fev_datasets` (revisione fissata), circa **650 MB** in tutto, mai scaricati finché l'utente non preme Download. Le definizioni dei task (colonna target, covariate note in anticipo, solo passate, statiche, orizzonte, finestre, stagionalità) vengono da fev-bench, non da me: non ho scritto a mano descrizioni dei dataset che non potessi verificare.
- **Verifica fatta da `scripts/build_catalog.py`** (rieseguibile, legge da Hugging Face senza salvare nulla, 1 minuto): lunghezza esatta di **ogni** serie, schema, e su un campione di 30 serie valori mancanti, asse temporale (la frequenza che davvero combacia: alcuni mensili stanno a inizio mese, per 37 voci la cartella non indica la frequenza (21 senza cartella, 16 «boomlet» con un numero al posto della frequenza) e quindi è dedotta dai dati), serie costanti, colonne di testo. Esito: nessun buco temporale nei campioni, nessuna serie costante; valori mancanti significativi in alcuni (rohlik vendite 14%, uci_air_quality 14%, walmart 34% nelle covariate note); tre colonne note in anticipo sono testo (festività) e diventano interruttori sì/no; 35 task sono multivariati (fino a 100 target); 7 file hanno più di 2000 serie (m5: 30.490, hermes: 10.000), quindi campionate.
- **Download sicuro:** conferma esplicita con dimensione e nota sulla licenza (le schede dichiarano «other»: valgono i termini delle fonti pubbliche), avviso per i file grandi (LOOP_SEATTLE 209 MB, m5 giornaliero 83 MB, kdd_cup 79 MB), verifica del **SHA-256** pubblicato da Hugging Face (provato su 4 file: tutti coincidono) e scarto del file se differisce; Stop e avanzamento nel pannello Attività; rimozione dalla pagina.
- **Integrazione:** i dataset scaricati compaiono nei selettori di Playground ed Esperimenti (gruppo «Online benchmarks»), con «max series» regolabile (campione con seme fisso: stesse serie ogni volta, 200 di default) e l'indicazione di horizon e finestre usati da fev-bench; lettura in streaming (un file da 30.000 serie non entra mai tutto in memoria); la contaminazione è «possibile» per tutti (dati di addestramento non dichiarati). Anche i file già scaricati dallo Showcase vengono riconosciuti.
- **Provato end-to-end** con Timestral-5M: previsione e backtest su `favorita_transactions_1D` (covariata nota «holiday» da testo a flag, covariata passata «oil_price»). 86 test (nuovi: integrità del catalogo, etichette uniche, lettore con buchi/testo/campionamento riproducibile, hash errato scartato, API, un vero download da 133 KB marcato `online`).
- **Non ancora fatto:** parte Monash di `chronos_datasets` e GiftEval (secondo e terzo livello del piano), sottoinsiemi con cataloghi propri; campo per un repository personalizzato.

### Correzione del menu del dispositivo e nuove etichette (2026-10-27)
- **Bug segnalato dall'utente:** con la pagina scorsa anche di pochi pixel, il menu CPU/GPU spariva appena si muoveva il cursore sulle opzioni e la pagina si spostava. **Causa trovata nel codice:** ogni opzione evidenziata (a ogni passaggio del mouse) chiamava `scrollIntoView`, che sposta anche la pagina quando c'è anche solo un pixel di margine; lo scorrimento della pagina chiudeva il menu. **Correzione in `web/js/select.js`** (vale per tutti i menu dell'app): si scorre solo l'elenco interno, e la chiusura per scorrimento scatta solo se il pulsante si è davvero spostato (più di 2 px). Verificato dal vivo: pagina scorsa di 5 px, menu aperto, cursore sulle opzioni, scelta della GPU: il menu resta aperto, la pagina non si muove, la scelta vale.
- **Etichette più semplici** nel solo menu del dispositivo: «CPU», «GPU · RTX 5060 Ti» su Windows e Linux, «GPU · Apple M5» su Mac (nome del chip invece di «Apple GPU (Metal)»); «GPU 1 · …», «GPU 2 · …» solo se ci sono più GPU; «GPU (unavailable)» se non disponibile. Gli identificativi tecnici (`cuda:0`, `mps`) e le specifiche complete restano nella pagina System e nei risultati delle esecuzioni, che devono dire con precisione dove sono girati.

### Notifiche delle attività, riprogettate (2026-10-27)
- **Una notifica per lavoro**, al posto di un unico riquadro fitto: icona di stato (quadratini animati in corso, spunta verde, avviso rosso, orologio in coda, croce se fermato), titolo in linguaggio naturale («Downloading», «Benchmarking», «Running a backtest», «Downloaded», «Backtest failed»…), il dettaglio (modello, dataset) sotto, e l'unica azione che serve.
- **Avanzamento onesto:** barra e «3 of 13» quando i passi sono noti, una barra che scorre quando non lo sono (i download), tempo trascorso; **nessuna stima di «tempo rimasto»**, perché i passi di un benchmark costano molto in modo diverso e il numero sarebbe sbagliato. Un lavoro in coda non mostra barra: dice che aspetta.
- **Comportamento:** le notifiche concluse spariscono da sole (12 s, 6 s se fermate) con una sottile barra di conto alla rovescia che si ferma al passaggio del mouse o alla tastiera; gli errori restano finché non li chiudi, con il messaggio completo espandibile; **Stop è riconosciuto subito** («Stopping…», poi «Stopped»: prima il lavoro fermato spariva senza dire nulla). Con più notifiche compare un'intestazione con «2 running · 1 finished», «Clear» e il pulsante per ridurre tutto a una pillola; sul telefono una sola scheda, le altre dietro «N more». Rispetta «riduci animazioni» e usa `aria-live`.
- **Un lavoro in coda che si ferma esce dalla coda subito** (prima aspettava il proprio turno, cioè la fine del lavoro davanti, mostrando «Stopping…» per minuti). Test aggiunto.
- **I pulsanti portano davvero dove dicono, e solo se non ci sei già:** ogni lavoro dichiara dove sta il suo esito (`link`): un download di modello → la scheda di quel modello nella pagina Models (`models?focus=timestral-209m`), un dataset → la sua riga nel catalogo (la lista si allarga fino a mostrarla), un benchmark → la sezione dei risultati in System, una misura dello Showcase → la scheda dello scenario giusto (`showcase?tab=retail`), un backtest concluso → la sua run. La pagina di arrivo scorre sull'elemento e lo evidenzia per un attimo. Se sei già nella pagina del lavoro il pulsante non c'è (resta solo Stop). Etichette specifiche: «Show the model», «Show the dataset», «See the results», «Open the run», «Go to Showcase».
- **Difetto trovato mentre lo provavo:** l'app teneva due iscrizioni separate alla route e a volte una pagina riceveva la route precedente al primo disegno (la scheda dello Showcase non si apriva sullo scenario richiesto): ora c'è una sola iscrizione.
- **Verificato dal vivo** (scuro e chiaro, 375 px senza scorrimento orizzontale): in coda, in corso, concluso, fermato, errore, e i quattro collegamenti. Nota: gli screenshot del pannello del browser integrato, catturati subito dopo un caricamento, mostrano le schede semitrasparenti perché le animazioni nel pannello partono in ritardo; lo stato reale (opacità 1, animazione conclusa) è stato verificato dal DOM.

### Conferme nello stile dell'app (2026-10-27)
- **Sostituiti tutti i `confirm()` nativi del browser** (5 punti: rimozione di un dataset caricato, di un file del catalogo, dei pesi di un modello, di una run dall'elenco e dal dettaglio) con `confirmDialog()` in `web/js/dialog.js`: titolo in chiaro, una frase che dice cosa succede davvero («the file you uploaded stays in data/user/», «You can download the weights again later», «This cannot be undone»), pulsante di conferma con il verbo preciso («Delete weights», «Delete run», «Remove») in rosso pieno, sfondo sfocato e scurito, scheda dal basso su telefono.
- **Più sicuro e più utile del `confirm()`:** il focus parte su «Cancel» (la scelta innocua), Tab resta dentro il dialogo, Esc e un clic fuori annullano, la pagina dietro non scorre e il focus torna al pulsante che l'ha aperto; l'azione viene eseguita *dentro* il dialogo (mostra «Deleting…», non si può chiudere nel frattempo) e **se fallisce l'errore compare nel dialogo con «Try again»** (prima l'errore di una cancellazione andava perso senza alcun avviso).
- **Provato dal vivo:** rimozione reale di un dataset (elenco aggiornato, file davvero rimosso), Esc, clic fuori, trappola del Tab, stato occupato, errore e nuovo tentativo, scuro/chiaro e 375 px senza scorrimento orizzontale. Test di regressione: nessun `confirm/alert/prompt` nativo nel codice dell'interfaccia e il dialogo è montato e usato da ogni azione distruttiva. 92 test.

### Secondo modello: ricerca e integrazione di Chronos-2 (2026-10-27)
**Metodo della ricerca.** Ho incrociato quattro fonti, privilegiando quelle primarie: (1) i **file dei risultati pubblicati** di fev-bench nel repository `autogluon/fev` (Apache-2.0, risultati di 43 modelli sugli stessi 100 task del nostro catalogo): ho **ricalcolato io la classifica** per i 30 che mi interessavano con il metodo di fev (punteggio di abilità = 1 − media geometrica dell'errore relativo al seasonal naive, risultati mancanti trattati come «non meglio della baseline»); il metodo riproduce al decimale i numeri che Mistral pubblica per Timestral-28M (SQL 48,42, MASE 37,27), e quelli di t0-gamma, TimesFM-3, Chronos-2 e TiRex-2 riportati da un blog terzo; (2) l'**API di Hugging Face** per scarichi degli ultimi 30 giorni, «mi piace» e licenza di ogni checkpoint; (3) l'**API di GitHub** per stelle, licenza e attività dei repository; (4) PyPI per dipendenze e versioni di Python. Le classifiche (fev-bench e GIFT-Eval) sono **inviate dai produttori dei modelli e non riprodotte da terzi**; su GIFT-Eval le prime posizioni sono ormai sistemi «agenti» o «router» costruiti sopra altri modelli, non modelli singoli, quindi non l'ho usata per scegliere.

**Classifica ricalcolata su fev-bench (100 task; abilità SQL in %, più alto è meglio):**

| Modello | Abilità tutti i task | Abilità task con covariate (42) | Vittorie a coppie | Licenza | Scarichi HF (API) |
|---|---|---|---|---|---|
| TimeH2O-v1 (H2O.ai, uscito il 26/10/2026) | 50,8 | 51,4 | 87,7% | «other» | 133 |
| **Timestral-209M** | 50,4 | 50,7 | 85,2% | Modified MIT (limite di ricavi) | – |
| Timestral-123M | 50,2 | 50,6 | 84,0% | Modified MIT | – |
| t0-gamma (The Forecasting Company, ~2 settimane) | 49,4 | 49,3 | 81,1% | Apache-2.0 | 26.731 |
| TimesFM-3.0 (Google, 331M) | 49,0 | 49,0 | 74,1% | **non commerciale** | 882.239 |
| Timestral-28M | 48,4 | 47,7 | 75,1% | Modified MIT | – |
| **Chronos-2 (Amazon, 120M)** | **47,3** | 47,0 | 64,6% | Apache-2.0 | **13.318.281** |
| TimesFM-2.5 (200M) | 46,7 | 47,8 | 53,4% | Apache-2.0 | 1.835.016 |
| Timestral-5M | 46,2 | 44,5 | 63,3% | Modified MIT | – |
| TiRex-2 (NX-AI, 82M) | 45,5 | 44,8 | 59,7% | Apache-2.0 | 6.525 |
| Toto-2.0 (313M / 1B / 2,5B) | 44,2 / 44,4 / 44,4 | **38,7 / 38,9 / 38,8** | ~60% | Apache-2.0 | 86.160 (313M) |
| Moirai-2.0 small | 40,2 | 37,1 | 35,9% | CC-BY-NC-4.0 | 265.317 |
| Chronos-Bolt (base) | 38,9 | 35,9 | 33,0% | Apache-2.0 | 1.764.176 |
| Sundial base (128M) | 33,9 | 29,1 | 20,3% | Apache-2.0 | 40.584 |
| Seasonal naive (riferimento) | 0 | 0 | 6,3% | – | – |

Cautele: Moirai-2.0 ha **28 dei 100 task nei propri dati di addestramento** (colonna «trained on this dataset» dei file dei risultati), TimesFM-2.5 ne ha 10, Toto-1.0 e FlowState 8; Timestral, Chronos-2 e TimesFM-3 zero. Le differenze fra i primi sei sono piccole (pochi punti di abilità) e le incertezze non sono pubblicate.

**Perché Chronos-2 come secondo modello** (criteri: licenza permissiva, popolarità reale, installabile nel nostro ambiente, **stesse capacità di Timestral** per un confronto onesto, accuratezza di vertice):
1. **Il più usato in assoluto** fra i modelli di previsione (13,3 milioni di scarichi secondo l'API di Hugging Face, di norma degli ultimi 30 giorni; 6.200 stelle sul repository) e fra i primi per accuratezza (settimo su questa classifica, dopo tre Timestral, t0-gamma, TimesFM-3 e TimeH2O: **il migliore con licenza Apache-2.0 fra quelli con una storia**).
2. **Capacità di input molto vicine a quelle di Timestral** (manca solo la covariata categorica): serie multivariate, covariate solo passate e note in anticipo, **gli stessi 21 livelli di quantile**: intervalli e punteggi sono confrontabili senza interpolazioni (solo la variante «small», addestrata su 13 livelli, viene interpolata e lo dice).
3. **Installazione pulita:** `chronos-forecasting` 2.3.2 su PyPI (Python ≥ 3.10, `torch` ≥ 2.2, `transformers`, `accelerate`) senza conflitti con il nostro `torch` 2.11 e Python 3.13; funziona su CPU, CUDA (la variante small differisce dalla CPU di 1,5·10⁻⁷) e, in linea di principio, MPS (non verificato).
4. **Dimensioni sensate:** 120M (478 MB) e 28M (112 MB) come i nostri 123M e 28M, quindi il confronto è anche di costo.

**Candidati scartati o rimandati, con il motivo:** *TimesFM-3.0* è il secondo candidato per accuratezza ma ha una **licenza non commerciale** che vieta l'uso in produzione e le opere derivate (va bene per prove locali, non per un progetto che potrebbe essere condiviso): rimandato, da decidere con l'utente; *TimesFM-2.5* è Apache ma più debole e con 10 task nei dati di addestramento; *TiRex-2* (Apache, 82M) è giovane (6.500 scarichi) e il pacchetto `tirex-ts` pubblicato è di giugno 2026, forse senza TiRex-2; *Toto-2.0* va male proprio dove contano le covariate (38,7 contro 47–51) e il pacchetto `toto-ts` **fissa `torch==2.7.0`, `numpy==1.26.4`, `pandas==2.2.3`**, incompatibili con il nostro ambiente; *Moirai-2.0* è CC-BY-NC e con il 28% del banco nei dati di addestramento; *TimeH2O* e *t0-gamma* sono vertici di classifica ma di giorni o settimane di vita, a dichiarazione del produttore e senza storia (t0-gamma Apache-2.0: ottimo candidato per più avanti); *Sundial*, *Chronos-Bolt*, *PatchTST* sono nettamente più deboli.

**Cosa è stato fatto nel codice:**
- **`TorchAdapter`** (`models/torch_adapter.py`): la parte comune a ogni modello su dispositivo torch (scelta CPU/GPU, ripiego sulla CPU con avviso, rilascio memoria, tempi sincronizzati). Timestral ora lo usa (nessuna modifica di comportamento: tutti i test dei modelli invariati) e **aggiungere un modello richiede solo due metodi** (`_load_on`, `_forecast_once`) più un manifesto per checkpoint.
- **`models/chronos.py`** + `manifests/chronos-2.toml` e `chronos-2-small.toml`: traduce una richiesta (storia, covariate passate, covariate note divise in parte passata e parte futura) nel formato di Chronos-2, richiede i nostri 21 livelli, ordina i quantili se si incrociano (e conta quante volte), svolge in modo autoregressivo gli orizzonti oltre 1024 passi, ignora le opzioni di Timestral. Il download dei pesi è ora generico (`snapshot_download`).
- **Interfaccia:** la pagina Models mostra una sezione per famiglia (con la scheda di ogni modello: strati, teste, patch, contesto, ingressi dedotti dalle capacità, benchmark pubblicati con la fonte); i menu dei modelli (Playground, Esperimenti, benchmark, Showcase) sono raggruppati per famiglia; **«Compare with»** ora offre, oltre alle baseline, gli altri modelli scaricati: eseguiti sulle stesse finestre con le stesse metriche. Pagina About: licenza di Chronos-2.
- **Provato dal vivo, GPU RTX 5060 Ti:** backtest a 4 modelli (Timestral-28M, Chronos-2, Chronos-2 Small, seasonal naive) su `favorita_transactions_1D` (covariate note e passate) e su una serie sintetica: **tutti conclusi**; MASE su traffico dei supermercati 0,521 (Timestral-28M) contro 0,509 (Chronos-2); su serie sintetica 0,272 contro 0,266 (differenze piccole, nessuna conclusione generale da 20 serie e 3 finestre). **Showcase con Chronos-2**, effetto misurato dei fattori sulle vendite Rossmann (40 negozi, 10 finestre): **Chronos-2 −31%, Chronos-2 Small −39%** (Timestral-28M −29%, 209M −41%): apertura, promozioni e festività contano, le vacanze scolastiche no (come con Timestral).
- **Test:** 6 nuovi (manifesti e ordine delle famiglie, divisione delle covariate, limite di contesto, quantili incrociati e avvisi, messaggio se i pesi mancano, previsione reale con la variante small).

### Terzo e quarto modello: TimesFM 3.0 e TimesFM 2.5 (2026-10-28)
- **Decisione dell'utente:** integrare entrambi, compreso TimesFM 3.0 nonostante la licenza non commerciale. Il testo di quella licenza (v1.0) consente per i pesi solo uso non commerciale e non in produzione (prove, valutazioni, ricerca), vieta la distribuzione e l'uso dei risultati in decisioni commerciali, consegne a clienti o prodotti a pagamento (le previsioni stesse non sono «derivati»); l'uso commerciale è permesso da Google solo tramite i suoi servizi cloud (BigQuery ML). **Come è reso visibile:** avviso giallo sulla scheda del modello, licenza nel suo nome, riga in «Known issues & license», paragrafo nella pagina About, e **finestra di conferma esplicita prima del download** («I understand, download») che riassume i limiti: senza conferma non parte nessun download (verificato dal vivo: rimozione, pulsante Download, finestra, conferma, scaricamento di 1,3 GB in pochi secondi).
- **Pacchetto:** `timesfm` 3.0.2 (PyPI, Apache-2.0) aggiunto **senza l'extra `torch`** (il nostro `torch` arriva dagli extra del progetto) e senza `jax`. Fornisce sia TimesFM 3.0 (`timesfm3`) sia 2.5.
- **TimesFM 3.0** (331M, 1,3 GB, contesto 15.360): serie univariate e multivariate, covariate solo passate e note in anticipo native, mediana + 9 decili; usa `TimesFM3Evaluator` con le impostazioni ufficiali di benchmark (media simmetrica e non-negatività attive, mappate sulle nostre opzioni, e divisione automatica delle variabili).
- **TimesFM 2.5** (231M con la testa dei quantili, 925 MB, contesto fino a 16.384): solo univariato; **le covariate sono ignorate e il risultato lo dice** (il pacchetto le offre solo con una regressione esterna, XReg, che richiede JAX). Particolarità risolte: il pacchetto sceglie da solo la GPU CUDA (ora si rispetta la scelta CPU/GPU dell'utente), `torch.compile` disattivato (su Windows serve un compilatore C++), ricompilazione solo quando cambiano contesto o orizzonte.
- **Livelli di quantile diversi (9 decili contro 21): confronto onesto.** SQL e WQL sono medie sui livelli, quindi non confrontabili fra modelli con livelli diversi. Ora, quando in un backtest i livelli non coincidono, **SQL, WQL e i punteggi di abilità sono calcolati per tutti i modelli sui livelli in comune** (nota in ogni risultato); i valori sui livelli propri restano come `sql_own_levels` / `wql_own_levels`; la copertura al 90% e 98% c'è solo per chi prevede quelle code (i decili danno solo l'80%). Lo Showcase disegna per i decili la fascia al 40% al posto del 50% e lo dice nella legenda; se il modello ignora i fattori, un avviso spiega che l'effetto misurato è zero per costruzione.
- **Provato dal vivo (RTX 5060 Ti):** parità CPU/GPU entro 1,2·10⁻⁶ per entrambi, GPU da 2,6 a 10 volte più veloce della CPU. **Backtest a cinque modelli** (Timestral-28M, Chronos-2, TimesFM 3.0, TimesFM 2.5, seasonal naive; 20 negozi, 3 finestre, orizzonte 28, con covariate) sul traffico dei supermercati: **MASE 0,521 / 0,509 / 0,480 / 0,486 / 0,727**; su una serie sintetica (8 serie): 0,272 / 0,266 / 0,271 / 0,275 / 1,421: i quattro modelli sono vicinissimi (pochi dati: nessuna conclusione generale), seasonal naive molto indietro. **Showcase, effetto dei fattori sulle vendite Rossmann (40 negozi, 10 finestre, GPU):** TimesFM 3.0 **−39,2%** (apertura −14,8%, promozioni −15,7%, festività −19,0%, vacanze scolastiche −1,4%), Chronos-2 −30,7%, Chronos-2 Small −38,6%; TimesFM 2.5 0% (ignora i fattori).
- **Test:** 8 nuovi (manifesti, licenze e avvisi, traduzione della richiesta per TimesFM 3.0 e 2.5 con modelli di prova, limite di contesto con le covariate tagliate allo stesso modo, colonna della media scartata, ricompilazione solo quando serve, messaggio se i pesi mancano, punteggio sui livelli in comune e nessun cambiamento quando coincidono, previsione reale con i due checkpoint se presenti). 106 test in totale.

### Quinto modello: t0-gamma, dopo una verifica di validità e sicurezza (2026-10-28)
**Il dubbio dell'utente:** t0-gamma è in cima a fev-bench ma è giovanissimo; vale la pena aggiungerlo rispetto alle alternative, ed è sicuro? Ho verificato prima di scrivere codice.

**Chi è e quanto è maturo.** Modello di The Forecasting Company (azienda con organizzazione GitHub dal 2023, repository `tfc-t0` dal 9/6/2026, 73 stelle, 13 versioni del pacchetto da giugno, fra i contributori un ingegnere di Hugging Face noto per modelli di previsione). Il **modello è stato pubblicato il 12/10/2026 (16 giorni fa)**, 26.731 scarichi, 24 «mi piace»; le due versioni precedenti (alpha, beta) sono descritte in un articolo (arXiv 2609.24559, 21/9/2026); **gamma non ha un articolo proprio**, solo la scheda del modello. Nel pacchetto e nell'articolo si riconosce il debito verso il codice aperto di Toto (Datadog) e Chronos-2 (Amazon), citato nel file NOTICE.

**Sicurezza, verificata sui fatti.**
- **Pesi:** un solo file `model.safetensors` (formato che non può contenere codice eseguibile), nessun file Python nel repository del modello, nessun `auto_map` né `trust_remote_code`, licenza Apache-2.0, non protetto da accesso.
- **Pacchetto `tfc-t0` 0.6.0:** ruota Python pura (75 KB, nessun codice compilato né script di installazione), il cui **SHA-256 coincide con quello pubblicato da PyPI**. Ne ho letto tutto il codice (circa 230 KB) cercando chiamate di rete, processi esterni, `eval`/`exec`, `pickle`, `torch.load`, variabili d'ambiente, telemetria: **nessuna** (solo import pigri di moduli propri e la lettura del file di configurazione; la rete serve unicamente a scaricare i pesi dall'hub se si passa un nome di repository, e noi carichiamo da cartella locale). Dipendenze nuove: `jaxtyping` e `wadler-lindig` (stesso autore, molto usate), `rotary-embedding-torch` (MIT, molto usata). Versione fissata esattamente (`tfc-t0==0.6.0`) e hash nel `uv.lock`: un aggiornamento richiede una nuova revisione.

**Validità dei risultati dichiarati.** Il punteggio di abilità su fev-bench (49,42) è nel file dei risultati pubblicato nel repository ufficiale `autogluon/fev` (richiesta di modifica #199 degli autori, accettata il 13/10/2026) e **lo ricalcolo identico** dai risultati (SQL 49,42; MASE 37,99); il valore di t0-beta (46,65) coincide con quello dell'articolo, quindi i loro numeri sono coerenti. Nessuna riga segnata come «addestrato su questo dataset» (ma è una dichiarazione degli autori). **Due cautele scoperte leggendo il codice della loro valutazione:** (1) sui task *senza covariate* hanno previsto **gruppi di 64 serie non correlate insieme** («cross-inference», informazione condivisa fra le serie): è una modalità legittima ma diversa dal prevedere ogni serie da sola, che è l'unica che questa app può usare senza far trapelare il futuro (in un backtest finestre diverse della stessa serie finirebbero nello stesso gruppo); (2) i dati di addestramento sono descritti solo come «dataset pubblici selezionati più serie sintetiche», **senza elenco**, e l'articolo non affronta la contaminazione.

**Verifica indipendente (la mia).** Stesse previsioni indipendenti per serie, finestre mie: 18 task del catalogo (12 senza covariate, 6 con), GPU, 12 serie per task, 3 finestre, contesto limitato a 2048 passi per tutti, SQL sui livelli in comune. **Abilità media sul seasonal naive: Timestral-209M 39,3 · t0-gamma 38,7 · Chronos-2 38,6 · TimesFM 3.0 38,0 · TimesFM 2.5 32,5.** Vittorie per task: Timestral-209M 8, TimesFM 3.0 4, t0-gamma 3, Chronos-2 3. **Lettura onesta: t0-gamma sta nel gruppo di testa insieme a Timestral, Chronos-2 e TimesFM 3.0 ma non lo supera** (sulla classifica pubblicata batte Chronos-2 di 2 punti, qui di 0,1: dentro il rumore di un campione così piccolo); con le covariate (6 task) è un po' dietro Chronos-2 e Timestral (43,5 contro 45,9 e 44,5), senza covariate è con Timestral e davanti a Chronos-2 (36,1 contro 34,6).

**Perché aggiungerlo comunque.** È il miglior modello con licenza Apache-2.0 nella classifica (49,4 contro 47,3 di Chronos-2 e 45,5 di TiRex-2) ed è nel gruppo di testa anche nella mia verifica; usa **gli stessi 21 livelli di quantile** e le stesse covariate di Timestral (confronti diretti, senza interpolazioni); sicurezza verificata. In questo strumento serve proprio a confrontare, non a eleggere un vincitore. Va però presentato per quello che è: **giovane, con un risultato dichiarato dagli autori, con dati di addestramento non elencati**: la scheda del modello e le note lo dicono.

**Integrazione.** `models/t0.py` + `manifests/t0-gamma.toml` sulla base `TorchAdapter`. La traduzione di una richiesta segue la costruzione pubblicata dagli autori nella loro valutazione (Apache-2.0): ogni variabile è una serie a sé con le stesse covariate, le covariate note vanno nella regione da prevedere, quelle solo passate sono righe «storiche» mancanti nell'orizzonte, con le quattro correzioni raccomandate (rimozione dei vuoti iniziali, media simmetrica, non-negatività, arrotondamento degli interi). Contesto limitato a 8192 passi (come nella loro valutazione), orizzonti oltre 1024 in modo autoregressivo, covariate note mancanti nell'orizzonte lette come 0 dal modello (avviso). Parità CPU/GPU 1,5·10⁻⁷. **Test:** 6 nuovi (manifesto e cautele, costruzione delle righe per ogni serie con modello di prova, tetto di contesto e suddivisione, avviso sulle covariate mancanti, messaggio senza pesi, previsione reale con il checkpoint se presente); 112 test in totale.

### Monash e GIFT-Eval nel catalogo online (2026-10-28)
**Cosa c'è ora:** 187 dataset (fev-bench 100, GIFT-Eval 55, archivio Monash 32), con le stesse garanzie di prima: download solo su richiesta, controllo SHA-256, conferma di licenza e dimensione, nota di contaminazione, marchio «also in» quando due benchmark condividono gli stessi dati.

**Come sono stati costruiti.** A differenza di fev-bench, questi due non si possono ispezionare a distanza (file Arrow e zip di testo non hanno un indice con le lunghezze), quindi `scripts/build_catalog_ext.py` scarica ogni file una volta, lo legge per intero (numero di serie, lunghezze, valori mancanti, serie costanti, asse temporale), lo cancella e registra lo SHA-256 pubblicato da Hugging Face. Revisioni fissate: GiftEval `30841734ac`, monash_tsf `58aafbe271`, regole di GIFT-Eval `38d068b769`. Totale circa 1,2 GB (GIFT-Eval) e 0,6 GB (Monash).

**Regole dei task.** GIFT-Eval: orizzonte per frequenza (M 12, W 8, D 30, H 48, minuti 48, secondi 60; M4 ha i suoi), finestre = ceil(10% della serie più corta / orizzonte) con tetto 20, e per i dati sub-giornalieri anche i termini medio e lungo (10× e 15×). Monash: orizzonte dichiarato nel file, altrimenti il valore predefinito del loader di Hugging Face (indicato nella scheda: «il file non ne dichiara»).

**Problemi trovati leggendo i dati (non fidandomi dei metadati).**
- Le cartelle di GIFT-Eval contengono anche copie `cache-*.arrow` lasciate da `datasets.map`: `electricity/15T` risultava 1110 serie invece di 370. Ora i file dati si prendono da `state.json`.
- `jena_weather` e `jena_weather/10T` hanno gli stessi byte: elencato una volta.
- `dataset_info.json` di `restaurant` dichiara 809 serie, i file e l'articolo 807: uso i file.
- Le date di partenza settimanali di GIFT-Eval non cadono sempre sull'ancora della frequenza: la frequenza settimanale segue il giorno reale di partenza.
- Le serie multivariate (ETT, Jena, Bitbrains) sono salvate come liste a dimensione fissa: convertite al volo.

**Sovrapposizioni.** Riconosciute per numero di serie e di osservazioni identici, solo fra benchmark diversi (es. `hospital` è nei tre, `solar` settimanale in tre, M-DENSE e SZ-taxi fra fev e GIFT). Verificate a mano: tutte corrette.

**Contaminazione.** Sono dati pubblici molto presenti nei corpora di pre-addestramento (il corpus di Chronos `autogluon/chronos_datasets` contiene dataset Monash) e i dati di Timestral non sono elencati: i punteggi sono indicazioni, non prova di abilità fuori campione. Lo dice la scheda di ogni voce.

**Verifica.** 10 dataset campione scaricati davvero (hash controllato), letti e passati in un backtest con il seasonal naive; pagina controllata nel browser. **Test:** 8 nuovi, 120 in totale.

### Classifica integrata: più modelli su un intero benchmark (2026-10-28)
**Cosa fa.** La pagina **Leaderboard** fa girare più modelli su tutti i dataset di un benchmark del catalogo (fev-bench, GIFT-Eval o Monash, filtrabili per dimensione del file) e li ordina. Ogni modello viene caricato una volta sola e prevede le stesse finestre di ogni dataset, con orizzonte e finestre del compito del benchmark (finestre limitate a 3, campione fisso di 20 serie per dataset, stesso tetto di contesto per tutti: 2048). Il seasonal naive gira sempre come riferimento.

**Come si calcola il punteggio.** Per ogni dataset l'errore del modello è la media su serie e finestre; il punteggio è 100·(1 − media geometrica, sui dataset, del rapporto errore del modello / errore del seasonal naive), con rapporti limitati fra 0,01 e 100 perché un solo dataset non domini: è lo stesso schema di fev-bench. Si calcola su SQL (ordinamento principale) e MASE, con numero di vittorie e tempo totale, per tutti i dataset e per gruppo. **Contano solo i dataset che tutti i modelli in classifica hanno completato**: un modello che non accetta serie multivariate non viene confrontato su un insieme più facile, e i dataset esclusi sono elencati con il motivo. I modelli con livelli di quantile diversi sono confrontati sui livelli condivisi.

**Sicurezza e onestà.** Se mancano dataset, la pagina li elenca con la dimensione totale e li scarica dopo conferma (licenza e verifica SHA-256 come sempre); il pulsante di avvio resta bloccato finché non sono scaricati, oppure si può scegliere di usare solo quelli già presenti. Un dataset troppo corto per il compito non ferma la corsa: viene registrato come non contato. Un avviso ricorda che i dati pubblici possono essere stati visti dai modelli e che 20 serie per dataset danno un risultato rumoroso: è un confronto alla pari fra modelli, non una prova di abilità fuori campione. Se si preme Stop non si salva nulla (il processo di calcolo viene interrotto).

**Prova reale.** 4 modelli (Timestral-209M, Chronos-2, TimesFM 3.0, t0-gamma) su 7 piccoli dataset di GIFT-Eval e Monash, GPU, 36 secondi. Risultato (solo un campione piccolo, da leggere come indicazione): t0-gamma +27,2% di abilità SQL, Timestral-209M +25,4%, TimesFM 3.0 +24,7%, Chronos-2 +19,7% sul seasonal naive: tutti e quattro vicini, coerente con la verifica precedente. **Test:** 5 nuovi (punteggio e limiti, regola dei dataset comuni, corsa completa con soli baseline e dataset troppo corto, archivio, rifiuti dell'API); 125 in totale.

**Salvataggio parziale (2026-10-28).** La classifica viene salvata dopo ogni dataset (scrittura atomica). Se si preme Stop, il calcolo si interrompe o il server si ferma, i dataset già completati restano: la corsa compare come «stopped» (Stop dell'utente) o «interrupted» (qualunque altra interruzione, anche un riavvio: lo stato si giudica alla lettura, senza riscrivere i file, così un secondo server non rovina le corse di un altro). La pagina mostra un avviso con quanti dataset su quanti pianificati sono stati eseguiti e la classifica conta solo quelli; una corsa ancora in corso si aggiorna da sola ogni 4 secondi. Le corse complete non vengono mai declassate. Prova reale: corsa da 10 dataset fermata a metà, 4 dataset conservati e classificati. **Test:** 2 nuovi (checkpoint e interruzione, stato «interrupted» via API); 127 in totale.
