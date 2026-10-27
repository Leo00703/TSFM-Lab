# Resoconto pre-sviluppo — Timestral Sandbox

Data: 2026-10-27 · Stato: **Fasi 0, 1 e 2 completate (§12); prossima: Fase 3**. Repository: `tsfm-lab` (privato)

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
- L'inferenza gira **nello stesso processo del server** (thread), non in un processo worker separato come previsto: i job sono serializzati, ma un crash di PyTorch fermerebbe anche il server. Resta da fare.
- **Dataset online** (Hugging Face) e badge di contaminazione per i dataset pubblici: Fase 4. Per ora i file caricati sono marcati "unknown".
- L'upload di file è coperto dai test API, ma non l'ho provato cliccando nel browser (gli strumenti non consentono di scegliere un file).
- Linux e macOS non testati; su Mac la GPU (MPS) resta disattivata per il bug upstream #6.
- ALTMistral e logo funzionano solo in locale (file ignorati da git); senza di essi l'interfaccia ripiega su Inter.

### Prossimi passi
Fase 3: pagina di confronto tra run, ulteriori viste e rifiniture di visualizzazione. Fase 4: dataset online, badge di contaminazione, preset delle covariate. Fase 5: processo worker per l'inferenza, controllo di conformità del brand, CI su Windows/Linux/macOS, secondo adapter di prova (es. Chronos-2).
