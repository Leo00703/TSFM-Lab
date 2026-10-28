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

### Prima corsa vera della classifica: GIFT-Eval, 5 modelli (2026-10-28)
**Cosa:** 49 dataset di GIFT-Eval sotto i 50 MB (249 MB scaricati, SHA-256 verificato), 5 modelli su GPU, 20 serie per dataset, al massimo 3 finestre, contesto 2048 per tutti, covariate attive; 8 minuti. Tutti i 49 dataset contati.

| Modello | Abilità SQL | Abilità MASE | Vittorie |
|---|---|---|---|
| t0-gamma | +35,0% | +31,3% | 16 |
| Timestral-209M | +34,5% | +30,2% | 16 |
| TimesFM 3.0 | +33,4% | +29,2% | 5 |
| Chronos-2 | +32,5% | +28,2% | 8 |
| TimesFM 2.5 | +29,5% | +25,4% | 4 |

**Lettura onesta.** I primi quattro sono entro 2,5 punti: sul campione usato (20 serie per dataset) la differenza fra t0-gamma e Timestral (0,5 punti) è dentro il rumore, e i dati di GIFT-Eval possono essere stati visti da alcuni modelli (Timestral non dichiara i suoi). Con covariate (8 dataset) Timestral-209M è primo (44,1 contro 43,0 di Chronos-2, 42,6 TimesFM 3.0, 42,3 t0-gamma); senza (41 dataset) è primo t0-gamma (33,5), poi Timestral (32,5) e TimesFM 3.0 (31,4). TimesFM 2.5, che ignora le covariate, è ultimo come atteso.

**Bug trovato dalla corsa.** Nella prima corsa `kdd_cup_2018` (orario) risultava «non contato» con un errore oscuro: TimesFM 2.5 andava in crash quando il contesto di una serie era interamente mancante (il pacchetto vuole usare zero, ma controlla male un array vuoto). Ora l'adattatore sostituisce quel contesto con zeri e lo segnala; inoltre la classifica distingue «troppo corto» da «un modello è fallito». Corsa rifatta da capo: 49 su 49. 128 test.

### Seconda corsa vera: fev-bench, 5 modelli (2026-10-28)
**Cosa:** 97 compiti di fev-bench sotto i 50 MB (su 100; 206 MB scaricati, SHA-256 verificato), 5 modelli su GPU, 20 serie per compito, al massimo 3 finestre, contesto 2048 per tutti, covariate attive; 15 minuti e 35 secondi, tutti i 97 contati.

| Modello | Abilità SQL | Abilità MASE | Vittorie |
|---|---|---|---|
| Timestral-209M | +40,8% | +35,1% | 37 |
| t0-gamma | +40,2% | +34,8% | 21 |
| TimesFM 3.0 | +39,9% | +34,4% | 18 |
| Chronos-2 | +37,7% | +31,7% | 10 |
| TimesFM 2.5 | +36,2% | +30,6% | 11 |

**Con covariate (40 compiti):** Timestral-209M 43,1 · t0-gamma 42,7 · TimesFM 3.0 41,6 · Chronos-2 40,2 · TimesFM 2.5 38,4. **Senza (57):** Timestral 39,2 · TimesFM 3.0 38,6 · t0-gamma 38,5 · Chronos-2 35,8 · TimesFM 2.5 34,6.

**Lettura onesta.** Su fev-bench Timestral-209M è primo per abilità e, soprattutto, per vittorie (37 compiti su 97 contro 21 di t0-gamma), ma il margine sul secondo e sul terzo è di 0,6-0,9 punti: con 20 serie per compito resta dentro il rumore, mentre il distacco da Chronos-2 e TimesFM 2.5 (3-4,6 punti) è più netto. Attenzione ai limiti: (1) il campione è piccolo e il contesto è limitato a 2048 passi per tutti; (2) fev-bench è un benchmark pubblico e Timestral non dichiara i suoi dati di addestramento; (3) la classifica ufficiale di fev-bench usa tutti i dati e metodi leggermente diversi (t0-gamma in testa), quindi questo non la smentisce né la conferma. Con GIFT-Eval (corsa precedente) l'ordine dei primi due era invertito di mezzo punto: i due modelli sono alla pari sui dati usati.

**Errore trovato.** La prima partenza è fallita subito perché le voci di fev-bench non avevano un titolo pronto (solo quelle di GIFT-Eval e Monash lo avevano): ora lo costruisce `hub_catalog.run_entries`, con un test che prova tutte le 187 voci. 129 test.

### Sesto modello: TiRex-2 (NXAI), dopo la verifica di validità e sicurezza (2026-10-28)
**Chi è.** Modello ricorrente (xLSTM, non Transformer) di NXAI GmbH (Linz) e del gruppo di Hochreiter; 82,5 M di parametri (38,4 M attivi in modalità univariata), un solo checkpoint per serie singole e multivariate, covariate passate e note in anticipo. Repository `NX-AI/TiRex-2` creato il 16/6/2026, **non protetto da accesso**, licenza Apache-2.0 per pesi e codice; articolo arXiv 2607.01204.

**Sicurezza, verificata sui fatti.** Pesi in `model.safetensors` (330 MB), che l'app scarica da solo insieme a `config.json` (il repository contiene anche un `model.ckpt`, che è un file pickle, e una copia ONNX: **non vengono scaricati**; il pacchetto stesso, per il checkpoint pickle, avvisa che «può eseguire codice arbitrario» e lo carica con `weights_only=True`). Pacchetto `tirex-2` 0.4.0: ruota Python pura (78 KB), **SHA-256 uguale a quello di PyPI**; ho cercato nel codice chiamate di rete, processi esterni, `eval`/`exec`, `pickle`, variabili d'ambiente: **nessuna** (solo il caricamento dei pesi dall'hub, che noi evitiamo passando una cartella locale). Dipendenze nuove: `flashrnn` (Apache, ruota pura), `mlstm_kernels`, `narwhals`, `ninja`, `matplotlib` e, su Windows, `triton-windows`; versione fissata e hash nel `uv.lock`.

**Un'avvertenza di licenza che non si vede dalla scheda.** `mlstm_kernels`, una libreria su cui il modello gira, è sotto la **NXAI Community Licence** (derivata da quella di Llama 3): libera per ricerca e per organizzazioni sotto i 100 milioni di euro di ricavi annui; sopra quella soglia, se la si incorpora in un prodotto o servizio commerciale serve una licenza commerciale di NXAI, e chi redistribuisce deve mostrare «Built with technology from NXAI». Per un banco di prova locale non cambia nulla, ma è scritto nelle note del modello e nel README, come per TimesFM 3.0.

**Validità dei risultati dichiarati.** Nel file di risultati pubblicato su `autogluon/fev` TiRex-2 ha abilità 45,5 su fev-bench: **è sotto Timestral-209M (50,4), t0-gamma (49,4), TimesFM 3.0 (49,0) e Chronos-2 (47,3)**, e anche sotto TimesFM 2.5 (46,7), t0-beta (46,7) e Timestral-5M (46,2); sopra solo TiRex-1 (42,7). Non è quindi un candidato a «migliore»: lo aggiungo perché è piccolo (82 M), veloce e ha una famiglia architetturale diversa (ricorrente), utile per confronti. Dati di addestramento: LOTSA e i dataset di Chronos, due corpora pubblici che coincidono in parte con quelli dei benchmark; esistono varianti «decontaminate» (TiRex-2-g, -gp, -f) che non uso.

**Integrazione.** `models/tirex.py` + `manifests/tirex-2.toml` sulla base `TorchAdapter`. Ogni elemento diventa un gruppo di variabili nativo (target, covariate passate, covariate future), i vuoti restano vuoti (li maschera il modello), contesto limitato agli ultimi 2048 passi, orizzonti oltre 320 in più passate fatte dal modello. Nove quantili (decili 0,1-0,9, come TimesFM: confronti sui livelli condivisi); eventuali quantili incrociati vengono rimessi in ordine con un avviso. **Kernel:** i kernel CUDA fusi del pacchetto richiedono un compilatore (nvcc/Triton) che su Windows di solito manca, e `torch.compile` serve un compilatore C++: ho verificato che il modello costruito con i **kernel in puro PyTorch**, spostato poi sulla GPU, dà le stesse previsioni (CPU contro GPU: differenza relativa 5·10⁻⁷, 4,6·10⁻⁷ con covariate) e gira ovunque in modalità eager (0,3-0,8 s per chiamata su RTX 5060 Ti, da 3 a 6 volte più veloce della CPU). La GPU Apple non è verificata (nota nel modello). **Test:** 5 nuovi (manifesto con avvertenze, traduzione della richiesta, contesto/vuoti/quantili incrociati, messaggio senza pesi, previsione reale con il checkpoint); 134 test in totale.

**TiRex-2 nelle stesse prove (2026-10-28).** Corse dedicate con le stesse impostazioni e gli stessi campioni delle due precedenti (stesse finestre, stesso riferimento seasonal naive, quindi i punteggi sono confrontabili): **GIFT-Eval** 49 dataset, abilità SQL **+31,6%** (MASE +27,5%), 33 s; **fev-bench** 97 compiti, **+36,9%** (MASE +31,0%), 79 s. Nella classifica dei sei modelli è quinto in entrambe: su GIFT-Eval fra Chronos-2 (32,5) e TimesFM 2.5 (29,5); su fev-bench fra Chronos-2 (37,7) e TimesFM 2.5 (36,2). Coerente con il suo risultato pubblico (45,5, sotto i primi quattro): è il più veloce di tutti per tempo totale dopo Chronos-2, con solo 82 M di parametri. CI verde su Linux, Windows e macOS.

### Campo per un repository Hugging Face personalizzato (2026-10-28)
**Cosa fa.** Sulla pagina Modelli si può indicare un repository pubblico (per esempio un fine-tuning di un modello supportato) e scegliere l'adattatore della famiglia che lo esegue. Il repository viene prima ispezionato (pubblico, un solo `model.safetensors` più `config.json`, al massimo 4 GB, licenza, autore, popolarità, commit); poi serve la conferma esplicita «so che sono pesi di terzi non verificati».

**Perché è stretto e non «esegui qualsiasi modello».** Eseguire un modello arbitrario significherebbe eseguire codice arbitrario (pickle, `trust_remote_code`, file Python). Qui: (1) si scaricano **solo** `config.json` e `model.safetensors` (il formato safetensors non può contenere codice; gli altri file del repository, anche pickle o Python, non vengono mai scaricati e l'avviso li elenca); (2) si scarica **il commit esatto che è stato ispezionato** e la copia in cache è legata a quello (un `push` successivo non cambia ciò che gira; per i modelli di Timestral il caricamento passa dalla cartella locale, non dal nome del repository); (3) repository privati o protetti sono rifiutati (l'app non ha credenziali), come i checkpoint a più file o oltre 4 GB e i repository già presenti fra i modelli integrati; (4) l'identificatore è validato con un'espressione rigida prima di qualsiasi chiamata di rete.

**Cosa non si può verificare e viene detto.** Contesto, livelli di quantile e supporto delle covariate sono quelli del modello di base: giusti per un fine-tuning di quel modello, sbagliati per un'altra architettura (che fallirebbe al caricamento o darebbe risultati privi di senso). Qualità e provenienza dei dati restano a carico di chi aggiunge il modello. La scheda mostra «custom checkpoint», il commit, un avviso e il pulsante per rimuoverlo (cancella anche i pesi); funziona dove funzionano gli altri modelli (Playground, Esperimenti, Classifica). Elenco salvato in `data/user/custom_models.json`; il processo di calcolo lo rilegge prima di ogni lavoro, quindi un modello aggiunto è usabile subito.

**Prova reale.** `NX-AI/TiRex-2-fevbench` aggiunto come modello personalizzato su TiRex-2: ispezionato (commit `acd6153b8e`, 82,5 M di parametri, Apache-2.0, il file `model.ckpt` ignorato), scaricato (315 MB, solo i due file), eseguito sulla classifica accanto a TiRex-2. **Punteggi identici** (22,98 su 6 dataset di prova): coerente col fatto che quel repository contiene lo stesso checkpoint (stesso hash del file `model.ckpt`), quindi la via «personalizzata» esegue davvero i pesi scaricati e dà gli stessi numeri del modello integrato. Modello di prova poi rimosso (pesi cancellati). **Test:** 11 nuovi (ispezione e rifiuti, validazione, aggiunta/rimozione e sincronizzazione fra processi, scaricamento dei soli due file al commit fissato, endpoint API con conferma obbligatoria); 145 in totale. Corretto anche il numero di strati mostrato nella scheda di TiRex-2 («undefined × 512»).

### Rifinitura del catalogo (2026-10-28)
**Nomi delle variabili di GIFT-Eval.** GIFT-Eval salva le variabili dei dataset multivariati senza nome («variable 1…»). Per **ETT1 ed ETT2** (8 voci) ora hanno i nomi veri delle colonne (HUFL, HULL, MUFL, MULL, LUFL, LULL, OT), ma solo dopo averli verificati sui dati e non per memoria: a 15 minuti le sette variabili coincidono **valore per valore** con le due serie di ETT in fev-bench (che le nomina) e l'ordine è lo stesso; le versioni oraria e giornaliera sono **le somme** di quelle a 15 minuti, variabile per variabile, e la settimanale è la somma della giornaliera (verificato anche questo). Per gli altri multivariati (Jena, Bitbrains, BizITObs) non ho trovato una fonte verificabile dell'ordine: restano numerati e la scheda lo dice, invece di inventare nomi.

**Orizzonti di Monash.** Prima, dove il file non dichiara l'orizzonte, usavo il valore predefinito del loader di Hugging Face (48 per i dati orari, 30 per i giornalieri…). Ho letto il paper del repository (Godahewa et al., 2021, sezione degli esperimenti) e ora applico **la sua regola**: orizzonte originale della competizione per i dataset di competizione (KDD Cup 2018: 48 ore); per gli altri 12 mesi, 8 settimane (5 per solar settimanale, già nel file), 30 giorni e **una settimana per i dati semi-orari, orari e a 10 minuti** (168 per gli orari, 336 per i semi-orari, 1008 per i 10 minuti). Cambiano 20 voci (per esempio traffic orario da 48 a 168, solar a 10 minuti da 48 a 1008). Una settimana di dati al minuto o ai 4 secondi non è praticabile (e il paper non ha valutato quei quattro dataset): restano al valore predefinito del loader, e la scheda dice per ogni voce **da dove viene l'orizzonte** (file, loader, paper, competizione, predefinito). Ho letto il paper estraendo il testo dal PDF in un ambiente temporaneo, senza toccare le dipendenze del progetto. **Test:** 2 nuovi (nomi ETT e note sui numerati; orizzonti e fonti di Monash); 147 in totale.

### Corsa rigorosa: solo dataset con covariate, 100 serie, 6 modelli (2026-10-28)
**Impostazioni.** I 48 dataset con covariate fra quelli sotto i 50 MB (40 di fev-bench, 8 di GIFT-Eval; 1849 serie in tutto), **100 serie per dataset** (invece di 20: fino a 300 finestre per dataset), al massimo 3 finestre, contesto 2048 per tutti, covariate attive, GPU, tutti e 6 i modelli, **tutti i dataset contati**. Durata 54 minuti e 41 secondi.

| Modello | Abilità SQL | Abilità MASE | Vittorie | fev-bench (40) | GIFT-Eval (8) |
|---|---|---|---|---|---|
| Timestral-209M | +42,3% | +36,3% | 16 | 42,0 | 43,7 |
| t0-gamma | +41,9% | +35,4% | 11 | 42,0 | 41,4 |
| TimesFM 3.0 | +40,9% | +34,7% | 5 | 40,7 | 42,2 |
| Chronos-2 | +40,1% | +33,3% | 7 | 39,8 | 41,5 |
| TiRex-2 | +37,7% | +31,0% | 3 | 37,4 | 39,2 |
| TimesFM 2.5 | +37,0% | +31,1% | 6 | 36,9 | 37,4 |

**Quanto è affidabile l'ordine.** Ho ricampionato i 48 dataset (bootstrap con 2000 ripetizioni) per misurare l'incertezza dovuta alla scelta dei dataset: Timestral-209M risulta primo nel 62% dei ricampionamenti, t0-gamma nel 27%, TimesFM 3.0 nel 7%. **Differenza di abilità rispetto a Timestral-209M** (mediana e intervallo al 90%): t0-gamma −0,4 [−2,0; +1,1], TimesFM 3.0 −1,4 [−3,3; +0,6], Chronos-2 −2,3 [−4,2; −0,1], TiRex-2 −4,6 [−8,1; −1,1], TimesFM 2.5 −5,2 [−10,4; −0,4]. **Conclusione onesta:** su questi dataset Timestral-209M, t0-gamma e TimesFM 3.0 **non sono distinguibili** (gli intervalli includono lo zero); Chronos-2, TiRex-2 e TimesFM 2.5 sono **dietro in modo credibile** (gli intervalli escludono lo zero, Chronos-2 di poco). Il bootstrap misura solo l'incertezza sui dataset, non quella sulle serie campionate dentro ogni dataset (ora 100 invece di 20, quindi minore), e non cambia il fatto che i dati sono pubblici e che Timestral non dichiara i suoi dati di addestramento: l'ordine fra i primi tre non è una prova di abilità fuori campione. Coerente con le corse precedenti (GIFT-Eval, fev-bench): stessi primi tre, stessi ultimi tre. TimesFM 2.5, che ignora le covariate, è ultimo come atteso; TiRex-2 resta quinto per SQL, a pari merito con TimesFM 2.5 sul MASE (31,0 contro 31,1). Tempo totale: Chronos-2 e TimesFM 2.5 i più veloci, TimesFM 3.0 il più lento (712 s).

### Classifica completa: tutti e 10 i modelli sugli stessi 48 dataset (2026-10-28)
**Impostazioni.** Identiche alla corsa rigorosa (48 dataset con covariate sotto i 50 MB, 100 serie, al massimo 3 finestre, contesto 2048, covariate attive, GPU), ma con **tutti i 10 modelli dell'app** in un'unica corsa: aggiunti Timestral-123M, Timestral-28M, Timestral-5M e Chronos-2 small. Tutti i dataset contati. Durata 76 minuti e 39 secondi. Per farla ho dovuto alzare da 8 a 24 il numero massimo di modelli per corsa (limite dell'API, ora con test).

**Controllo di riproducibilità.** I 6 modelli già misurati hanno dato **gli stessi numeri della corsa precedente**: errori identici bit per bit per Timestral-209M, Chronos-2, TimesFM 3.0, TimesFM 2.5, TiRex-2 e il baseline; per t0-gamma la differenza massima è 2e-6 in valore relativo. Anche i tempi sono simili (per esempio TimesFM 3.0: 719 s contro 712 s), anche se la GPU era condivisa con altri programmi: la condivisione non ha alterato né i punteggi né, in modo visibile, i tempi.

| Modello | Abilità SQL | Abilità MASE | Vittorie | fev-bench (40) | GIFT-Eval (8) | Primo nel bootstrap | Differenza da Timestral-209M (90%) | Tempo (s) |
|---|---|---|---|---|---|---|---|---|
| Timestral-209M | +42,3% | +36,3% | 11 | 42,0 | 43,7 | 47% | — | 804 |
| Timestral-123M | +42,0% | +36,1% | 5 | 42,0 | 42,0 | 19% | −0,3 [−1,1; +0,7] | 501 |
| t0-gamma | +41,9% | +35,4% | 10 | 42,0 | 41,4 | 25% | −0,4 [−2,0; +1,3] | 372 |
| TimesFM 3.0 | +40,9% | +34,7% | 3 | 40,7 | 42,2 | 6% | −1,4 [−3,2; +0,6] | 719 |
| Chronos-2 | +40,1% | +33,3% | 5 | 39,8 | 41,5 | 1% | −2,2 [−4,2; 0,0] | 252 |
| Timestral-28M | +39,9% | +33,9% | 3 | 39,4 | 42,2 | 0% | −2,4 [−3,7; −1,0] | 173 |
| Chronos-2 small | +38,6% | +32,3% | 3 | 38,9 | 37,2 | 0% | −3,7 [−6,2; −0,8] | 61 |
| TiRex-2 | +37,7% | +31,0% | 1 | 37,4 | 39,2 | 1% | −4,5 [−7,9; −0,9] | 415 |
| Timestral-5M | +37,2% | +31,4% | 2 | 36,6 | 40,5 | 0% | −5,0 [−7,6; −2,2] | 72 |
| TimesFM 2.5 | +37,0% | +31,1% | 5 | 36,9 | 37,4 | 2% | −5,2 [−10,3; −0,7] | 193 |

**Cosa si può dire.** (1) **Timestral-209M, Timestral-123M e t0-gamma non sono distinguibili** (gli intervalli includono lo zero); TimesFM 3.0 è appena sotto e anch'esso non distinguibile da Timestral-209M. Timestral-123M è a 0,3 punti dal 209M con **il 62% del tempo di calcolo** (501 s contro 804 s): su questi dati la taglia più grande non porta un vantaggio dimostrabile. (2) Chronos-2 è dietro di poco (l'intervallo sfiora lo zero). (3) **Timestral-28M** (+39,9, 173 s) sta quasi al livello di Chronos-2 (+40,1, 252 s) e **batte Chronos-2 small** (+38,6, 61 s) con probabilità 81% nel bootstrap, ma l'intervallo della differenza (−1,3; +3,7) include lo zero, quindi non è dimostrato. (4) **Timestral-5M** (+37,2, 72 s) è al livello di TiRex-2 e TimesFM 2.5 con 1/6 e 1/3 del loro tempo: il più veloce dopo Chronos-2 small. (5) TimesFM 2.5 resta ultimo, come atteso: ignora le covariate. Le caveat precedenti restano valide: dati pubblici e possibile contaminazione (Timestral non dichiara i dati di addestramento), il bootstrap copre solo l'incertezza sulla scelta dei dataset e non quella sulle serie campionate, e i punteggi valgono per dataset piccoli e medi con covariate. **Non coperti:** dataset sopra i 50 MB, dataset senza covariate nella corsa a 100 serie, Monash nella classifica.

### Revisione di documenti, interfaccia, solidità e sicurezza (2026-10-28)
Fatta mentre girava la corsa senza covariate: nessun riavvio del server di lavoro, nessun uso della GPU (i test con `CUDA_VISIBLE_DEVICES=""`, l'interfaccia provata su un secondo server sulla porta 8766).

**Sicurezza e solidità del backend (letto tutto `app.py`, `jobs.py`, `worker.py`, store, caricamenti, modelli personalizzati).** Cose già a posto: nessun `pickle`/`eval`/`exec`/`trust_remote_code` nel codice, pesi solo `safetensors`, parser `.tsf` che non esegue nulla, controllo di `Host` e `Origin`, repository personalizzati validati con espressione rigida e scaricati al commit ispezionato, nessun `innerHTML` nell'interfaccia. Trovato e corretto:
1. **Identificatore di una corsa preso dall'URL senza controllo** (`RunStore`): con barre rovesciate (Windows) poteva puntare fuori da `runs/` (lettura di un `meta.json` altrove o cancellazione di una cartella che ne contiene uno). Non l'ho sfruttato, ma il percorso c'era: ora l'id deve avere esattamente la forma che l'app genera.
2. **Caricamento di file**: il corpo veniva letto tutto in memoria prima di controllare il limite di 100 MB. Ora si rifiuta subito (intestazione `Content-Length`) o appena il limite è superato; il tipo di file è controllato prima di scrivere; un file illeggibile non resta su disco; gli stessi byte con due nomi tengono un solo file; la lettura avviene in un thread, così un file grande non blocca il server.
3. **Cancellazioni sotto un lavoro in corso**: cancellare un dataset (catalogo o proprio) mentre una corsa lo usa l'avrebbe fatta fallire a metà; cancellare una corsa della classifica ancora in esecuzione la faceva ricomparire. Ora 409 con un messaggio chiaro.
4. **Intestazioni di sicurezza**: `Content-Security-Policy` (solo propri script, nessuno script inline, nessun altro sito), `X-Frame-Options: DENY`, `nosniff`, `Referrer-Policy`. Per questo lo script del tema è passato da inline a `web/js/theme-init.js`. Verificato nel browser: nessuna violazione, grafici e pagine funzionano. Escluse solo le pagine `/api/docs` (caricano il visualizzatore da una CDN).
5. **Esportazione CSV**: gli identificativi vengono dai file dell'utente; quelli che iniziano con `=`, `+`, `-`, `@` ora sono preceduti da un apice, così un foglio di calcolo non li esegue come formula.
6. **Scrittura non atomica** dello stato di una corsa della classifica (`finish`): ora temporaneo più rinomina.
7. **Pagina Sistema che andava in errore** con una build CUDA di PyTorch e `CUDA_VISIBLE_DEVICES=""` (`is_available()` vero con zero dispositivi): ora controlla anche `device_count()`, e un driver rotto non la fa più cadere.
**Non coperto (detto nel README):** nessun login su Tailscale (chiunque sulla tailnet usa l'app), un file costruito per sfruttare un difetto di `pandas`/`pyarrow`/`openpyxl`, chi ha accesso alla cartella. I test sono 10 nuovi (`tests/test_hardening.py`); 156 in totale.

**Interfaccia (ogni pagina letta, desktop, telefono e tema chiaro).** Corretti: numeri e date nel formato del browser invece che in inglese (`32.768`, `28 ott 2026`: ora `32,768`, `28 Oct 2026`); dimensioni in MiB etichettate «MB» (798 MB invece di 837: ora unità decimali, come Hugging Face e il README); nomi dei modelli grezzi (`timestral-209m`) nelle tabelle di Classifica ed Esecuzioni, ora i nomi veri, con «10 modelli» quando sono molti; nomi dei dataset (`entsoe`, `ecdc ili`, `bizitobs l2c` → `ENTSO-E`, `ECDC ILI`, `BizITObs L2C`) e frequenza nelle tabelle della classifica (prima tre righe «entsoe» identiche); «Attention heads: download to read» per TiRex-2, che è ricorrente e non ne ha (riga nascosta); «multivariate» per TimesFM 2.5, che le variabili le prevede una alla volta (ora lo dice); testi fissi «Timestral» dove il modello è qualunque (Esperimenti, Playground, Showcase); una pagina «Overview» morta (nascosta e senza link) rimossa; «Licenses/license» → «Licences/licence» come nel resto; nomi delle fonti («fev-bench», «GIFT-Eval») invece di `fev`, `gift`; durata «1 h 17 min» invece di «4599 s». La pagina Informazioni ora cita TiRex-2 e i checkpoint personalizzati, dice che i quantili sono 21 o 9 a seconda del modello e spiega il punteggio della classifica. **Telefono:** nessuno scorrimento orizzontale su nessuna pagina (375 px); tema chiaro leggibile.
**Visto e lasciato:** all'apertura dello Showcase partono due richieste di previsione che trovano il calcolo occupato (409) e riprovano da sole: nessun effetto, solo righe rosse nella console; il nome di alcuni dataset differisce nella maiuscola fra fonti diverse (`LOOP SEATTLE` / `LOOP Seattle`) perché è quello scritto dalle fonti.

**Documenti.** README: introduzione con tutti e 10 i modelli (TiRex-2 mancava), elenco delle pagine con Classifica e Informazioni, 187 dataset da tre fonti (prima diceva 100 e poi 187), TiRex-2 fra i modelli a nove decili, i due file di CI, nuove sezioni «Security» e «Development». Il comando `ruff` non è nel README perché il progetto non è pulito per `ruff` (95 segnalazioni, soprattutto di stile).

### Classifica sui dataset senza covariate: 127 dataset, 3 fonti, 10 modelli (2026-10-28)
**Impostazioni.** Tutti i dataset senza covariate sotto i 50 MB: 57 di fev-bench, 41 di GIFT-Eval, 29 di Monash (127). 100 serie per dataset, al massimo 3 finestre, contesto 2048, GPU libera, tutti e 10 i modelli, **tutti i 127 dataset contati**. Durata 31 minuti. Corsa `20261028-143109-e201`.

**Un difetto trovato dalla prima corsa (125 dataset su 127).** Due dataset Monash (orizzonte 336 e 1008) fallivano con un errore di forme: **TiRex-2 taglia in silenzio ogni orizzonte oltre 320 passi** (il pacchetto stampa solo un avviso) e l'avviso che avevo scritto io nell'adattatore («continua in più passaggi») era falso. Ora l'adattatore fa i passaggi: ognuno riparte dalla mediana del precedente (le covariate solo-passato, ignote nel futuro, passano come mancanti), con avviso e nota che gli intervalli successivi tendono a essere troppo stretti (misurato su una sinusoide: errore della mediana 0,02 → 0,06 e ampiezza dell'intervallo all'80% 0,17 → 0,10 dal primo all'ultimo passaggio). Controllati anche gli altri 9 modelli oltre il loro limite per passaggio: tutti restituiscono l'orizzonte completo. 2 test nuovi. La prima corsa resta nella cronologia (job `c5d4bfc3178a`).

| Modello | Abilità SQL (127) | Senza doppioni (107) | Abilità MASE | fev-bench (57) | GIFT-Eval (41) | Monash (29) | Differenza da Timestral-209M (90%) | Tempo (s) |
|---|---|---|---|---|---|---|---|---|
| Timestral-209M | +39,3% | +40,1% | +35,0% | 39,2 | 31,4 | 49,3 | — | 89 |
| Timestral-123M | +38,8% | +39,6% | +34,8% | 38,9 | 31,0 | 48,5 | −0,5 [−0,8; −0,2] | 64 |
| Timestral-28M | +38,2% | +39,1% | +34,2% | 38,0 | 30,6 | 48,4 | −1,0 [−1,4; −0,7] | 33 |
| TimesFM 3.0 | +38,0% | +38,7% | +33,8% | 38,5 | 30,5 | 46,7 | −1,3 [−2,8; 0,0] | 461 |
| t0-gamma | +37,8% | +38,3% | +34,5% | 38,4 | 32,3 | 44,1 | −1,6 [−4,5; +0,4] | 103 |
| Timestral-5M | +36,8% | +37,5% | +32,2% | 36,8 | 29,6 | 46,1 | −2,6 [−3,6; −1,7] | 24 |
| TimesFM 2.5 | +35,0% | +35,8% | +30,7% | 34,5 | 26,8 | 46,4 | −4,3 [−6,7; −2,1] | 269 |
| TiRex-2 | +34,9% | +35,6% | +28,4% | 35,8 | 28,1 | 42,3 | −4,4 [−6,3; −3,0] | 192 |
| Chronos-2 | +33,4% | +33,6% | +28,6% | 36,1 | 28,8 | 34,3 | −6,2 [−12,7; −2,8] | 46 |
| Chronos-2 small | +31,8% | +31,7% | +27,9% | 34,9 | 26,8 | 32,2 | −8,1 [−14,7; −4,6] | 22 |

«Senza doppioni» (colonne a destra, bootstrap su 2000 ricampionamenti): 43 dei 127 dataset compaiono in più fonti (per esempio in fev-bench e in GIFT-Eval); qui un dataset condiviso pesa una volta sola, 107 distinti. Il bootstrap è su questi 107.

**Cosa si può dire.** (1) **Timestral-209M è primo in modo più netto che con le covariate**: primo nell'86% dei ricampionamenti, davanti a Timestral-123M di 0,5 punti con un intervallo che esclude lo zero (piccola ma coerente), e a 28M di 1,0. t0-gamma resta **non distinguibile** da lui (−1,6 [−4,5; +0,4]); TimesFM 3.0 è al limite (−1,3 [−2,8; 0,0]). (2) **Le taglie piccole di Timestral reggono**: 28M (33 s) sta sopra TimesFM 2.5, TiRex-2 e Chronos-2; 5M (24 s) sta sopra TimesFM 2.5 e TiRex-2 e batte Chronos-2 small (probabilità 100% nel bootstrap, differenza +5,5 [+2,1; +12,1]) con lo stesso tempo di calcolo. (3) **TimesFM 3.0 batte TimesFM 2.5** (+3,0 [+1,6; +4,3]), che ignora le covariate ma qui non ne servono. (4) **Chronos-2 e Chronos-2 small in fondo, ma per un solo dataset**: su `monash.bitcoin` Chronos-2 (entrambi), t0-gamma e TiRex-2 danno errori enormi (fino a 1e14 volte il baseline per Chronos-2; i Timestral e i TimesFM no), il rapporto viene limitato a 100 e questo da solo sposta i punteggi (per TiRex-2 l'errore SQL è indefinito e il dataset non entra nel suo punteggio, quindi non è penalizzato). **Senza `monash.bitcoin`** l'ordine è: Timestral-209M 39,4 · t0-gamma 39,1 · Timestral-123M 38,9 · Timestral-28M 38,4 · TimesFM 3.0 38,2 · Timestral-5M 36,9 · Chronos-2 36,0 · TimesFM 2.5 35,1 · TiRex-2 34,9 · Chronos-2 small 34,4: t0-gamma risale al secondo posto e Chronos-2 supera TimesFM 2.5 e TiRex-2. **Non so ancora se la causa è nei dati (serie con valori mancanti o scale estreme) o nei modelli**: è una questione aperta, da indagare prima di fidarsi dell'ordine fra i modelli sotto il quinto posto. (5) Per fonte l'ordine cambia: GIFT-Eval è il benchmark più difficile (punteggi sotto 33 per tutti), Monash il più facile (fino a 49).

**Confronto con la classifica delle covariate** (altri dataset, quindi solo indicativo): Timestral-209M primo e Timestral-123M in alto in entrambe; senza covariate t0-gamma scende dal 3° al 5° posto e Chronos-2 dal 5° al 9° (per `monash.bitcoin`, vedi sopra), i Timestral piccoli salgono. **Limiti.** Dataset sotto i 50 MB (restano esclusi gli 8 più grandi senza covariate, e quelli con covariate sopra i 50 MB); dati pubblici e dati di addestramento di Timestral non dichiarati; il bootstrap copre solo l'incertezza sulla scelta dei dataset; tempi misurati a GPU libera, una sola passata.

### Il caso `monash.bitcoin` e la classifica senza covariate corretta (2026-10-28)
**Causa.** Il dataset non contiene 18 prezzi ma 18 grandezze della rete Bitcoin; una di queste, `hashrate`, va da 5e5 a 2e20. Un quadrato di 1e20 supera il massimo del float32, e i modelli che calcolano le statistiche in float32 sbagliavano: **Chronos-2** e **TiRex-2** davano su quella serie errori di 1e16-1e17 volte il baseline, **TimesFM 2.5** dava NaN, **t0-gamma** restituiva valori non finiti per tutto il gruppo (le 18 serie viaggiano insieme: una sola serie estrema rovinava le altre 17, che l'adattatore sostituiva con zero). Timestral e TimesFM 3.0 non erano colpiti. Verificato: dividendo la serie per la sua media tutti i modelli tornano normali (errore scalato 5-7 contro 6,5 del baseline), cioè non erano dati sporchi.

**Correzione.** Nell'adattatore di base (`rescale_extreme`): ogni serie di target o di covariata con valori fuori da 1e-12...1e15 viene divisa per il suo massimo assoluto prima del modello e la previsione è moltiplicata di nuovo, con un avviso nel risultato. I modelli sono per costruzione indifferenti alla scala; le serie dentro l'intervallo non vengono toccate (nessun calcolo: i risultati restano identici bit per bit). Controllo: fra i 175 dataset scaricati solo `monash.bitcoin` ha valori fuori intervallo, quindi tutte le altre righe di tutte le classifiche restano valide. 6 test nuovi (162 in totale), fra cui uno con i modelli veri su una serie che arriva a 1e24.

**Classifica corretta** (corsa `20261028-152221-8b6d`: stessi 127 dataset, tutti contati, 32 minuti; **sostituisce la tabella della corsa precedente**, che resta nella cronologia):

| Modello | Abilità SQL (127) | Senza doppioni (107) | Abilità MASE | fev-bench (57) | GIFT-Eval (41) | Monash (29) | Differenza da Timestral-209M (90%) | Tempo (s) |
|---|---|---|---|---|---|---|---|---|
| Timestral-209M | +39,3% | +40,1% | +35,0% | 39,2 | 31,4 | 49,3 | — | 92 |
| t0-gamma | +39,0% | +39,6% | +35,6% | 38,4 | 32,3 | 48,5 | −0,4 [−1,8; +0,8] | 106 |
| Timestral-123M | +38,8% | +39,6% | +34,8% | 38,9 | 31,0 | 48,5 | −0,5 [−0,8; −0,2] | 66 |
| Timestral-28M | +38,2% | +39,1% | +34,2% | 38,0 | 30,6 | 48,4 | −1,0 [−1,4; −0,7] | 34 |
| TimesFM 3.0 | +38,1% | +38,8% | +33,9% | 38,5 | 30,5 | 46,9 | −1,3 [−2,8; 0,0] | 474 |
| Timestral-5M | +36,8% | +37,5% | +32,2% | 36,8 | 29,6 | 46,1 | −2,6 [−3,6; −1,7] | 25 |
| Chronos-2 | +35,9% | +36,6% | +31,2% | 36,1 | 28,8 | 44,8 | −3,5 [−4,6; −2,4] | 48 |
| TimesFM 2.5 | +35,0% | +35,8% | +30,7% | 34,5 | 26,8 | 46,3 | −4,3 [−6,8; −2,1] | 282 |
| TiRex-2 | +34,8% | +35,5% | +31,1% | 35,8 | 28,1 | 41,7 | −4,5 [−6,4; −3,1] | 200 |
| Chronos-2 small | +34,4% | +34,8% | +30,6% | 34,9 | 26,8 | 43,1 | −5,3 [−6,5; −4,1] | 23 |

**Cosa si può dire (sostituisce le conclusioni della corsa precedente).** (1) In testa **Timestral-209M** (primo nel 72% dei ricampionamenti) e **t0-gamma** (25%), **non distinguibili** (−0,4 [−1,8; +0,8]); Timestral-123M è sotto di 0,5 punti in modo piccolo ma coerente, TimesFM 3.0 al limite (−1,3 [−2,8; 0,0]). (2) **Il crollo di Chronos-2 e di t0-gamma era un difetto nostro, non dei modelli**: t0-gamma torna secondo, Chronos-2 settimo e dietro i Timestral di 3,5 punti con un intervallo che esclude lo zero. (3) **Le taglie piccole**: Timestral-28M (34 s) sopra Chronos-2, TimesFM 2.5 e TiRex-2; Timestral-5M (25 s) sopra TimesFM 2.5 e TiRex-2 e davanti a Chronos-2 small di 2,7 punti [+1,7; +3,8], con lo stesso tempo di calcolo. (4) TimesFM 3.0 batte TimesFM 2.5 (+3,0 [+1,7; +4,4]). (5) Nessun dataset ha più un modello oltre 5 volte l'errore del baseline. Restano le cautele: dati pubblici e dati di addestramento di Timestral non dichiarati, bootstrap solo sulla scelta dei dataset, dataset sotto i 50 MB.

### I 12 dataset più grandi (da 50 MB in su): classifica e un'anomalia di TimesFM 2.5 (2026-10-28)
**Corsa** `20261028-222653-e620`: i 12 dataset del catalogo da 50 MB in su (1,7 GB, scaricati in 35 s), 8 senza covariate e 4 con covariate (`kdd_cup_2022_10T`, `m5_1D`, `gift.bitbrains_*`), 10 modelli, stesse impostazioni delle altre corse (100 serie, al massimo 3 finestre, contesto 2048). Tutti i 12 contati, 19 minuti. Con questa corsa **tutti i 187 dataset del catalogo sono stati misurati** da tutti i modelli (127 + 48 + 12).

| Modello | 12 dataset | Senza `kdd_cup_2022_10T` (11) | Differenza da Timestral-209M (90%) | Solo gli 8 senza covariate |
|---|---|---|---|---|
| TimesFM 2.5 | +49,9% | +44,2 | −1,8 [−3,8; −0,1] | 47,6 |
| Timestral-209M | +45,8% | **+46,0** | — | **48,5** |
| t0-gamma | +45,1% | +45,6 | −0,3 [−2,8; +1,7] | 48,1 |
| Chronos-2 | +44,9% | +45,4 | −0,6 [−2,4; +1,5] | 48,4 |
| Timestral-123M | +45,0% | +45,2 | −0,7 [−1,4; −0,3] | 47,7 |
| TiRex-2 | +44,6% | +45,1 | −0,9 [−2,9; +1,6] | 48,1 |
| Timestral-28M | +44,8% | +45,0 | −0,9 [−1,5; −0,4] | 47,8 |
| TimesFM 3.0 | +44,6% | +45,0 | −0,9 [−3,2; +1,1] | 47,8 |
| Timestral-5M | +43,6% | +43,6 | −2,3 [−4,6; −0,6] | 46,3 |
| Chronos-2 small | +41,7% | +41,6 | −4,3 [−8,0; −1,6] | 43,5 |

**L'anomalia.** Con tutti i 12 dataset TimesFM 2.5 risulta primo (+49,9), ma il vantaggio viene da un solo dataset: su `kdd_cup_2022_10T` (potenza di turbine eoliche, previsione a 48 ore) il suo errore è 0,15 volte quello del baseline, contro 0,55-0,61 di tutti gli altri nove modelli, cioè **quattro volte più basso di qualunque altro**. Controlli: (1) non dipende dalle covariate (stesso valore 0,30 con e senza; TimesFM 2.5 le ignora comunque), (2) non dipende dalla procedura (tutti i modelli passano dallo stesso codice, e il contesto finisce prima della finestra da prevedere), (3) vale in tutte e tre le finestre (0,34, 0,48 e 0,09 contro 1,34, 1,81 e 0,33 di Timestral-28M), (4) l'errore mediano di TimesFM 2.5 è in media 0,26 deviazioni standard del contesto e per il 31% delle serie sotto 0,1, mentre Timestral-28M sta a 1,06 e per nessuna serie sotto 0,1. Un errore così basso non si ottiene prevedendo la produzione di una turbina a 48 ore, e il dataset contiene turbine dello stesso parco sulle stesse date. **La causa non è nota**: un'ipotesi è che il modello abbia già visto questi dati in addestramento, ma non è dimostrata (i dati di addestramento di TimesFM 2.5 non sono pubblici) e l'analisi successiva sul profilo dell'errore lungo l'orizzonte non la conferma (vedi sotto). Va quindi **trattato come non confrontabile** e il dato resta nella cronologia ma non va usato per giudicare i modelli.

**Cosa si può dire.** (1) Senza quel dataset la graduatoria ricalca quella delle altre corse: Timestral-209M primo, t0-gamma, Chronos-2, TiRex-2, Timestral-123M e Timestral-28M raggruppati entro un punto, TimesFM 2.5 sotto, Timestral-5M e Chronos-2 small in fondo; in testa nessuna differenza dimostrabile (t0-gamma −0,3 [−2,8; +1,7]). (2) Sugli 8 dataset senza covariate Timestral-209M è primo con 48,5 ma Chronos-2 è a 0,1 punti: a quella scala e con 8 dataset non si distingue. (3) Solo 12 dataset: gli intervalli sono larghi e la corsa conferma, non sostituisce, le due classifiche principali (127 dataset senza covariate, 48 con covariate). (4) Su `monash.wind_farms_minutely` tutti i modelli stanno a 0,13-0,17 del baseline (il baseline sbaglia molto più di tutti) e su `monash.london_smart_meters` TiRex-2 deve fare la previsione in più passate (orizzonte 336 oltre i 320), come segnalato. Nessuna riga ha valori non finiti né errori; i modelli con covariate vedono gli avvisi già noti (`m5_1D`: covariata costante nel contesto mascherata).

**Nell'app (2026-10-28).** `kdd_cup_2022_10T` ora ha un avviso nel Catalogo ("results not comparable", con la spiegazione) e la Leaderboard lo lascia fuori dalla classifica per impostazione predefinita, con una nota in cima e un pulsante "Count it anyway"; nella tabella "Every dataset" la riga resta, marcata. L'elenco degli avvisi è `CAUTIONS` in `data/hub_catalog.py`; il ricalcolo senza i dataset segnalati lo fa l'API a ogni apertura, quindi vale anche per le corse già salvate.

### Altri dataset con un risultato fuori misura? Ricerca sulle tre classifiche (2026-10-28)
**Metodo.** Per ognuno dei 186 dataset con un punteggio utile (127 senza covariate, 48 con, 12 grandi, tutti i modelli) e per ogni modello: errore SQL diviso per la mediana dell'errore degli altri nove. La dispersione normale è minima (scarto robusto 0,04 in log2): quasi tutte le coppie stanno entro ±30% della mediana. Un valore sotto 0,5 è quindi un caso fuori dal normale, ma **non dice da sé la causa** (dati visti in addestramento, serie molto prevedibile per quel modello, difetto o semplice abilità).

**Casi sotto 0,6 volte la mediana degli altri.** `kdd_cup_2022_10T` TimesFM 2.5 0,27; `monash.oikolab_weather` TimesFM 2.5 0,34; `proenfo_gfc12` TimesFM 2.5 0,38; `proenfo_gfc17` TimesFM 2.5 0,45; `gift.solar.H` t0-gamma 0,49; `proenfo_gfc14` TimesFM 2.5 0,56; più `monash.sunspot` (Timestral-209M 0,38, e 123M/28M a 0,5) e `monash.wind_4_seconds` (Timestral-28M 0,52, 123M 0,60), dove però tutti i modelli sbagliano meno di 0,02 deviazioni standard: sono serie quasi perfettamente prevedibili e i rapporti, fra errori minuscoli, non hanno significato. **TimesFM 2.5 ha 5 degli 11 casi sotto 0,6 (di tutti i modelli) ma anche il rapporto medio più alto (1,07 volte la mediana degli altri)**: è molto bravo su pochi dataset di energia e meteo e nella media è il peggiore.

**Profilo dell'errore lungo l'orizzonte** (errore assoluto della mediana diviso per la deviazione standard del contesto, primo quarto contro ultimo quarto dell'orizzonte; ricalcolato su 10 dataset sospetti più `m5_1D` come controllo, con il modello favorito e due modelli di controllo). Un modello che ha memorizzato la serie dovrebbe avere un errore piatto e minuscolo anche alla fine dell'orizzonte. Non è quello che si vede: su `kdd_cup_2022_10T` l'errore di TimesFM 2.5 sale da 0,22 a 0,51 (×2,3) come quello dei controlli (da 0,6 a 2,0, ×3,3), solo che parte da tre volte più in basso; su `oikolab_weather` l'errore di TimesFM 2.5 non sale (da 0,25 a 0,20) mentre quello dei controlli quasi raddoppia (da 0,5 a 0,9), il caso più vicino a una memoria; su `solar.H` e sui `proenfo_gfc*` il profilo è simile a quello dei controlli. **Il profilo non permette di distinguere memoria da abilità**, quindi non lo uso come prova: la formulazione corretta è «risultato inspiegato», non «contaminazione». Una ricerca sul web dice solo che TimesFM 2.5 usa il corpus GiftEvalPretrain (fonte secondaria) e non cita né KDD Cup 2022 né GEFCom.

**Effetto sulle classifiche** (tolti i 2 dataset sospetti della classifica a 127 e i 3 `proenfo_gfc*` della classifica a 48): in testa non cambia nulla (Timestral-209M primo in entrambe; nella classifica a 127 t0-gamma e Timestral-123M si scambiano il secondo posto, differenze di 0,2 punti). Cambia TimesFM 2.5: −0,5 punti a 127 (da ottavo a nono) e **−5,1 punti a 48 (da 37,0 a 31,9, resta ultimo)**, perché quei tre dataset di GEFCom reggono da soli il suo risultato con covariate.

**Decisione.** Segnalo e lascio fuori dalla classifica solo `kdd_cup_2022_10T`, dove il rapporto è 0,27 e il dataset da solo capovolge la graduatoria dei 12. Gli altri non vengono esclusi: non c'è alcuna prova, escluderli penalizzerebbe un modello su una sola ipotesi e le conclusioni non cambiano. Restano documentati qui.

**Un'osservazione a margine.** `monash.solar_4_seconds` risulta contato nella classifica a 127 ma non contribuisce: la serie è costante nel contesto, l'errore di scala è nullo e SQL, MASE e WQL sono vuoti per tutti i modelli. Dei 127 dataset contano quindi 126. **Corretto**: la classifica conta solo i dataset su cui il baseline ha un errore utilizzabile, le corse già salvate vengono riconteggiate all'apertura (126 di 127, punteggi invariati) e il dataset compare in "Not counted" con il motivo.

### Dati di addestramento: cosa dichiarano gli autori, e la causa dell'anomalia di TimesFM 2.5 (2026-10-28)
**Cosa ho trovato.** I file dei risultati di fev-bench che ogni autore ha inviato (repository `autogluon/fev`, `benchmarks/fev_bench/results`, commit `ceb25ae`) hanno una colonna `trained_on_this_dataset`. **Gli autori di TimesFM 2.5 dichiarano 10 dataset come parte del suo addestramento**: `favorita_transactions_1D`, `favorita_transactions_1W`, `fred_md_2025--cee`, `fred_md_2025--macro`, **`kdd_cup_2022_10T`**, `m5_1D`, `m5_1W`, **`proenfo_gfc12`**, **`proenfo_gfc14`**, **`proenfo_gfc17`**. Gli altri otto con un file (i quattro Timestral, Chronos-2, TimesFM 3.0, t0-gamma, TiRex-2) dichiarano zero su 100 compiti; Chronos-2 small non ha un file proprio, quindi per lui non c'è nessuna dichiarazione. I quattro in grassetto sono **esattamente i dataset su cui TimesFM 2.5 aveva un errore da 3 a 4 volte più basso degli altri**.

**Correzione di quanto scritto prima.** (1) L'anomalia di `kdd_cup_2022_10T` **non è più «inspiegata»: è la sovrapposizione con l'addestramento, dichiarata dagli stessi autori**. Il profilo dell'errore lungo l'orizzonte non la mostrava da solo (l'errore cresceva come negli altri): la prova è la dichiarazione, non la forma dell'errore. (2) Avevo deciso di non escludere `proenfo_gfc12/14/17` «per mancanza di prove»: **ora le prove ci sono**, quindi sono esclusi insieme agli altri dichiarati. (3) La sovrapposizione dichiarata non dà sempre un vantaggio: su `m5_*`, `favorita_*` e `fred_md_*` TimesFM 2.5 sta alla pari o dietro (rapporto 0,95-1,36 rispetto alla mediana degli altri).

**Nell'app.** `DECLARED_TRAINING_DATA` in `data/hub_catalog.py` sostituisce l'avviso scritto a mano: un dataset dichiarato come addestramento di un modello classificato ha l'avviso nel Catalogo e, in una corsa che include quel modello, è escluso dalla classifica per impostazione predefinita (resta nella tabella, marcato; un pulsante lo include). Lo stesso ricalcolo vale per le corse già salvate e per l'elenco «Previous runs» (migliore e conteggio sono quelli della classifica senza i dichiarati). Una corsa senza TimesFM 2.5 non ha avvisi.

**Classifiche ricalcolate senza i dichiarati** (bootstrap su 2000 ricampionamenti dei dataset):
- **127 senza covariate** (1 dichiarato, `fred_md_2025--macro`; 125 restano): nulla cambia. Timestral-209M 39,4 (primo nel 67%), t0-gamma 39,1 (32%, −0,3 [−1,5; +0,8]), Timestral-123M 38,9, Timestral-28M 38,4, TimesFM 3.0 38,1, Timestral-5M 36,9, Chronos-2 36,2, TimesFM 2.5 35,2, TiRex-2 34,9, Chronos-2 small 34,6.
- **48 con covariate** (7 dichiarati; 41 restano): **cambia il vertice**. t0-gamma 41,1 (primo nel 43%), Timestral-209M 40,8 (28%; −0,3 [−2,1; +1,4]), Timestral-123M 40,7 (14%), Chronos-2 39,6, TimesFM 3.0 39,6, TiRex-2 39,0, Chronos-2 small 39,0, Timestral-28M 39,0, Timestral-5M 37,3, **TimesFM 2.5 32,2 (ultimo, −8,8 [−13,4; −4,8])**. I primi tre restano non distinguibili: il titolo «Timestral-209M primo» della classifica a 48 dipende dalla scelta di tenere o no i dichiarati, e va detto.
- **12 grandi** (2 dichiarati, `kdd_cup_2022_10T` e `m5_1D`; 10 restano): Timestral-209M 46,7, t0-gamma 46,3, Chronos-2 46,1, Timestral-123M 45,9, TiRex-2 45,8, Timestral-28M 45,7, TimesFM 3.0 45,7, TimesFM 2.5 44,9, Timestral-5M 44,3, Chronos-2 small 42,0.

**Timestral.** Ho letto l'annuncio di Mistral (`mistral.ai/news/timestral`), la scheda del modello e il repository: «la maggior parte dei dati di addestramento era sintetica», dati reali solo nella seconda fase (con contesto 4 volte più lungo), nessuna fonte reale nominata, nessuna analisi di sovrapposizione; Mistral afferma che nulla di ciò che viene valutato è stato visto nel pre-addestramento, ma è un'affermazione sul processo, non verificabile. Il rapporto tecnico è «in arrivo». Quindi la frase «Timestral non dichiara i suoi dati di addestramento» era troppo forte: dichiara che sono in prevalenza sintetici e non elenca la parte reale. Corretti i testi di Catalogo e README. **Non ho trovato nulla sui dati di Chronos-2 e dei Monash/GIFT**: per i dataset di GIFT-Eval e Monash nessuno dichiara niente, e due risultati fuori misura restano senza spiegazione e non esclusi (`monash.oikolab_weather` TimesFM 2.5 a 0,34 volte la mediana degli altri; `gift.solar.H` t0-gamma a 0,49).

**Ricontrollo e Chronos-2 (2026-10-29).** I file dei risultati di fev-bench riletti all'ultimo commit (`193e4fc`, 28 ottobre: aggiunge un modello nuovo, iAmTime) hanno **le stesse dichiarazioni**: 10 per TimesFM 2.5, zero per gli altri otto. Rapporto tecnico di Timestral: **non ancora pubblicato** (nessun risultato). **Chronos-2** (scheda e articolo arXiv 2510.15821): dati reali solo univariati, scelti dai corpora di Chronos e di GIFT-Eval Pretrain (Electricity, M4, Solar, Taxi, Uber TLC, Weatherbench, Wiki, USHCN, LargeST, Q-Traffic, Buildings 900K…); compiti multivariati e con covariate interamente sintetici; la quota di sintetico non è indicata. Gli autori escludono dal corpus le parti di *test* di ogni compito GIFT-Eval e segnalano una sovrapposizione parziale con le parti di *addestramento* di alcuni dataset GIFT-Eval; per fev-bench e Chronos Benchmark II dicono che nessun dataset è stato visto, senza descrivere un controllo. Conseguenza pratica: su GIFT-Eval i risultati di Chronos-2 su serie che stanno anche nel suo addestramento (finestre di test escluse) sono comunque da leggere con questa cautela; non cambia nessuna classifica e non c'è nulla da escludere per questo modello. Il DECLARED_TRAINING_DATA resta com'è.

### Pulizia con ruff (2026-10-29)
`ruff check .` ora passa senza avvisi (prima 130; in precedenza il README parlava di 95). 84 correzioni sicure automatiche (ordine degli import, `noqa` inutili, sintassi obsoleta); 24 avvisi sulla concatenazione implicita di stringhe nelle tuple di note sono stati risolti mettendo le parentesi, **verificando con l'albero sintattico (hash AST di tutti i 75 file Python) che il codice fosse identico**. A mano: `check=False` esplicito nelle 7 chiamate a `subprocess.run` (stesso comportamento), `noqa` motivati dove un `except … pass` è voluto (pulizia della cache, informazioni del sistema), argomenti predefiniti nelle funzioni definite in un ciclo (`B023`), qualche semplificazione (`Self` in `__enter__`, `min` al posto di `sorted()[0]`, un `if/else` con due rami uguali nel Showcase). **Trovato un difetto nei test:** `tests/test_isolation.py` conteneva le sue prime quattro prove due volte (un incidente di una modifica precedente, già nello storico): le prime copie venivano ridefinite e non giravano; rimosse le copie, ora il file ha 5 prove distinte. Non ho applicato `ruff format` (cambierebbe 66 file di stile senza correggere nulla). 165 test superati.

### Classifica sui propri dataset (2026-10-29)
Preparazione del passo «dati mai visti dai modelli», che aspetta di avere i dati: la Leaderboard ha ora la scheda **Your datasets**. Si scelgono uno o più file caricati nella pagina Datasets, un orizzonte unico (passi in avanti) e le stesse impostazioni delle altre corse; le finestre sono le ultime di ogni serie. Dietro: `SuiteBody.local_datasets` e `local_horizon`, `DatasetCatalog.suite_entries` (voci con fonte «mine», id del dataset locale), caricamento locale nel worker; l'API chiede l'orizzonte (422) e controlla che i dataset esistano (404); un dataset troppo corto per l'orizzonte finisce sotto «Not counted». Il gruppo della classifica si chiama «Your data». **Provata davvero**: file temporaneo con 3 serie giornaliere, due modelli sulla GPU, corsa completata, riga con etichetta «your data» e classifica corretta; file e corsa poi cancellati. Un dataset perfettamente periodico non viene contato (il baseline stagionale sbaglia zero, nessun rapporto): vale anche per i dati propri. 2 test nuovi, 167 in totale.

### Licenza del codice (2026-10-29)
Scelta dell'utente: **Apache-2.0**, intestata a Leonardo Galli (2026); la repo **resta privata**. Aggiunti `LICENSE` (testo ufficiale, hash SHA-256 verificato `cfc7749b…`), `NOTICE` (copyright e un elenco di ciò che la licenza non copre: pesi, dataset, librerie e font inclusi, marchi Mistral) e i campi `license`, `license-files` e `authors` in `pyproject.toml`; il README non dice più «licenza non ancora scelta». Pubblicare la repo resta una decisione a parte dell'utente.
