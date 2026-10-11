# Alpha Radar V18.4｜資料可追溯、信心分級、多週期選股整合版

**正式部署及本次修正：[`RELEASE_V18_4.md`](RELEASE_V18_4.md)**；驗證範圍：[`VALIDATION_STATUS_V18_4.md`](VALIDATION_STATUS_V18_4.md)。

原始目標不變：從台股完整可交易母體，以 **10 / 40 / 120 交易日**分別做短中長線排名；已完成日K與盤中核對行情、歷史案例、股票吸引力與買點/部位風控各自獨立。V18.4 把原預計 V18.1～V18.4 的資料品質、PIT、信心、排名穩健度、全球風險和介面工作整合，但**不宣稱完成真實歷史市場樣本外績效驗證**。

## 最快安裝

1. 將**完整包**內 `alpha_radar_v18_4/` 目錄下所有檔案放到 GitHub 專案根目錄並覆蓋同名檔案（保留 `.github/workflows`）。
2. Streamlit 入口 `app.py`；原 `FINMIND_TOKEN` 若使用，仍由 Streamlit Secrets 設定，勿寫入程式。
3. 執行 `python run_full_qa.py`；部署後手動按一次「背景更新日線模型」，才能建立 V18.4 三分數與時間證據。舊版 Snapshot 僅作暫時閱讀，勿誤當新模型。
4. 不上傳 `data/*.sqlite`、金鑰、`__pycache__`；但若需要長期 PIT 歷史回放，必須將 PIT SQLite 保存於真正**持久化**儲存空間。單純 Streamlit Cloud 本地磁碟可能重啟遺失，系統目前不具備外部儲存保證。

---

## 舊版沿革（以下僅供追溯）
# Alpha Radar V18.0 — 多因子透明選股、分週期評分與國際市場觀察室

**最新部署請先閱讀 [`RELEASE_V18_0.md`](RELEASE_V18_0.md)。**

V18.0 以 V17.5 為底盤，新增 15 項可解釋因子（13 項正向選股證據＋2 項買點風險）、短中長線獨立權重、產業共振、缺失資料標示與一頁式手機因子透視。選股與買點分開，不把夜盤下跌當成公司品質降低。保留原行情日期一致性、背景更新、手機斷線續看、國際市場快讀、推薦總表。

## 版本來源

# Alpha Radar V17.5 — 全球市場觀察室 / 手機斷線續看 / 精簡推薦總表

**最新版本功能與安裝方法請先看 [`RELEASE_V17_5.md`](RELEASE_V17_5.md)。**

本版保留 V17.4 資料健檢與 V17.1 日期同步框架，額外加入「最近美股指數點位與日期」、「重要數據比較判讀」、「新聞 RSS 摘錄與條件式台股影響直接閱讀」、「伺服器端背景日線掃描與跨連線讀取進度」及「視覺合併去重推薦總表」。

原模型的短／中／長線選股規則、歷史案例推估及買點防線仍按原定義運作。所有離線測試並非真實市場績效、即時行情或 Streamlit Cloud 實際速度的驗證。

---

# Alpha Radar V17.1 — Synced Market / Fast Brief

V17.1 is a production-integrity release on top of V17.0. It fixes mixed TWSE/TPEx completed-date risk, makes universe coverage explicit, adds horizon-specific final investability/evidence guardrails, translates global-market abbreviations into plain Traditional Chinese, and gives the user a useful global/news/previous-model briefing while the expensive canonical update runs.

Key invariants remain unchanged: short/mid/long raw selection weights are frozen; global news/context never changes stock-quality ranking; intraday executable decisions still require verified prices; Panic Reversal remains a downstream entry route rather than a selection factor.

See `DEBUG_REPORT_V17_1.md` and `VALIDATION_STATUS_V17_1.md` for the exact synchronization, coverage and QA contract.

---

# Alpha Radar V17.0 — Goal Complete / Full-Market Canonical

V17.0 is the first release whose production contract is explicitly aligned with the original goal: **at any query time, identify the best available Taiwan-stock candidates for short (10 sessions), mid (40 sessions), and long (120 sessions), then separately decide whether the current price is actionable.**

## Canonical production model

The normal UI no longer exposes candidate-count/history-length knobs that could silently change the answer. Production is fixed to:

- completed-day history: 5 years;
- universe: every liquid common stock with sufficient completed-day history;
- model family: full evidence;
- short/mid/long selection weights unchanged from the frozen core;
- global/US context affects execution risk and panic-reversal logic only, never stock-quality ranking.

## V17.0 changes

1. **Full liquid-universe baseline** — `candidate_size` no longer truncates production ranking. Every liquid stock with adequate history gets short/mid/long forecasts.
2. **Comparable breadth research** — official TWSE/TPEx market-wide revenue, latest financial, valuation and institutional evidence are loaded on one comparable scale for ranking. Rich historical research is loaded for leading names for display only and cannot change the published ranking.
3. **Missing evidence is neutral, not rewarded** — absent fundamental/flow evidence is neutral-imputed and receives a small bounded coverage penalty. Missing fields no longer inflate the remaining weights through re-normalization.
4. **Same-date core reuse** — if the completed-session date has not changed, the expensive historical/analog core is reused and only comparable research/ranking is refreshed.
5. **Full-market intraday discovery** — the one-minute scan quotes up to 2,500 liquid names and sends a bounded detail pool through official MIS price verification before BUY/SELL/HOLD execution scoring.
6. **Three entry routes remain separate** — Pullback, Continuation and Panic Reversal. Falling knives, structural breaks and near-limit-down situations remain blocked.
7. **Doctor consistency** — on-demand stock diagnosis uses the same comparable ranking evidence as the market selector; rich research is display detail, not an unfair ranking advantage.

## Time-of-day behavior

- **Pre-open:** latest completed-day canonical ranking + next-session watchlist + overnight global context.
- **Intraday:** completed-day ranking stays frozen; quotes refresh every 60 seconds; official MIS-verified execution prices drive buy/hold/sell decisions.
- **Post-close:** today's official EOD is used when complete; otherwise the last completed model remains authoritative and is clearly labeled pending.
- **Off-day:** latest completed session is reused; no fake daily candle is created.

## Core selection weights (unchanged)

| Horizon | Technical | Historical analog | Fundamental | Institutional flow |
|---|---:|---:|---:|---:|
| Short | 52% | 20% | 8% | 20% |
| Mid | 42% | 22% | 22% | 14% |
| Long | 34% | 22% | 34% | 10% |

Selection answers **what is worth owning**. Entry timing answers **whether this price is worth paying now**. Market/global regimes size risk; they do not rewrite selection quality.

## Verification

Run:

```bash
python run_full_qa.py
```

The bundled GitHub workflow `.github/workflows/qa_v17.yml` runs the same suite on push. A separate `validate_v17_2026_walkforward.py` performs leakage-controlled real-stock 2026 replay when external market data are reachable.

## Validation boundary

Offline engineering QA is complete. Real-market profitability is **not claimed** from the packaging container because external Taiwan market connectivity is unavailable there. The validator refuses to substitute synthetic data for real performance.

## Deployment

Upload the package source files to the repository; do **not** upload `__pycache__` or `.pyc` files. Existing price/research SQLite caches can be kept, but the first V17 update must build a new V17 dashboard snapshot because the model version changed.

## V17.3 — Mobile Recommendation Board

A new compact **推薦總表** board supports list filters (completed daily or next-session), horizon filters (short/mid/long), cross-horizon consensus (>=2 *distinct completed-day horizons*) and 3/5/8/15 picks per horizon. A Top 15 expandable summary and separately collapsed full comparison table reduce mobile scrolling. This release does not alter model weights or historical ranking logic.

See `DEBUG_REPORT_V17_3.md` and run `python test_v17_3_mobile_board.py`.
