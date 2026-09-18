/* Polymarket manual decision dashboard — gated pilot workflow.
 * No automated orders. Personal limits/positions stay in localStorage.
 */
const LS_RISK = "mlb_metrics_pm_risk_v1"
const LS_MANUAL = "mlb_metrics_pm_manual_v1"
const LS_BOARD_SEEN = "mlb_metrics_pm_board_generated_at"

let boardRows = []
let boardMeta = {}
let evaluation = null
let pilotReadiness = null

async function loadCSV(path){
  const response = await fetch(`${path}?t=${Date.now()}`, {cache: "no-store"})
  if(!response.ok) throw new Error(`Failed to load ${path}: ${response.status}`)
  const text = await response.text()
  return Papa.parse(text, {header:true, skipEmptyLines:true}).data
}

async function loadJSON(path){
  const response = await fetch(`${path}?t=${Date.now()}`, {cache: "no-store"})
  if(!response.ok) throw new Error(`Failed to load ${path}: ${response.status}`)
  return response.json()
}

function selectPmTab(tab){
  document.querySelectorAll("#pmTabs .tabButton").forEach(btn=>{
    btn.classList.toggle("active", btn.dataset.pmTab === tab)
  })
  document.querySelectorAll(".tabPanel").forEach(panel=>{
    panel.classList.toggle("active", panel.id === `tab-${tab}`)
  })
}

function fmtPct(v){
  if(v === undefined || v === null || v === "" || Number.isNaN(Number(v))) return "—"
  return `${(Number(v) * 100).toFixed(1)}%`
}

function fmtNum(v, digits=3){
  if(v === undefined || v === null || v === "" || Number.isNaN(Number(v))) return "—"
  return Number(v).toFixed(digits)
}

function fmtAge(sec){
  if(sec === undefined || sec === null || sec === "" || Number.isNaN(Number(sec))) return "—"
  const s = Number(sec)
  if(s < 60) return `${s.toFixed(0)}s`
  if(s < 3600) return `${(s/60).toFixed(1)}m`
  return `${(s/3600).toFixed(1)}h`
}

function stateClass(state){
  if(state === "Paper candidate") return "state-paper"
  if(state === "Game started") return "state-started"
  if(state === "Pass—pilot paused") return "state-started"
  return "state-pass"
}

function riskConfigured(){
  const r = loadRisk()
  if(!r) return false
  return Number(r.bankroll) > 0
    && Number(r.max_loss) > 0
    && Number(r.per_bet) > 0
    && Number(r.daily) > 0
    && Number(r.same_game) > 0
    && Number(r.same_team) > 0
    && Number(r.max_loss) <= Number(r.bankroll)
}

function loadRisk(){
  try{ return JSON.parse(localStorage.getItem(LS_RISK) || "null") }catch{ return null }
}

function saveRiskLimits(){
  const payload = {
    bankroll: Number(document.getElementById("r_bankroll").value),
    max_loss: Number(document.getElementById("r_max_loss").value),
    max_affordable_loss: Number(document.getElementById("r_max_loss").value),
    per_bet: Number(document.getElementById("r_per_bet").value),
    per_bet_exposure_limit: Number(document.getElementById("r_per_bet").value),
    daily: Number(document.getElementById("r_daily").value),
    daily_exposure_limit: Number(document.getElementById("r_daily").value),
    same_game: Number(document.getElementById("r_game").value),
    same_game_exposure_limit: Number(document.getElementById("r_game").value),
    same_team: Number(document.getElementById("r_team").value),
    same_team_exposure_limit: Number(document.getElementById("r_team").value),
    saved_at: new Date().toISOString(),
    income_target_not_used: true,
  }
  if(!(payload.bankroll > 0) || !(payload.max_loss > 0) || !(payload.per_bet > 0) || !(payload.daily > 0) || !(payload.same_game > 0) || !(payload.same_team > 0)){
    document.getElementById("riskStatus").textContent =
      "All five limit families are required: bankroll, max affordable loss, per-bet, daily, and correlated (same-game + same-team)."
    renderReadiness()
    return
  }
  if(payload.max_loss > payload.bankroll){
    document.getElementById("riskStatus").textContent =
      "Max affordable loss cannot exceed bankroll."
    return
  }
  localStorage.setItem(LS_RISK, JSON.stringify(payload))
  const verdict = (pilotReadiness?.evidence_for_limited_real_money_pilot?.verdict)
    || boardMeta.evidence_verdict
    || "unknown"
  document.getElementById("riskStatus").textContent =
    `Saved. Stake guidance still requires a passing evidence gate (currently: ${verdict}). Stakes never auto-increase after losses.`
  renderReadiness()
  renderBoard()
}

function clearRiskLimits(){
  localStorage.removeItem(LS_RISK)
  document.getElementById("riskStatus").textContent = "Cleared. Using normalized paper units only."
  renderReadiness()
  renderBoard()
}

function loadManual(){
  try{ return JSON.parse(localStorage.getItem(LS_MANUAL) || "[]") }catch{ return [] }
}

function saveManualPosition(){
  const row = {
    id: `${Date.now()}_${Math.random().toString(16).slice(2,8)}`,
    market_id: document.getElementById("m_market_id").value.trim(),
    game_pk: document.getElementById("m_game_pk").value.trim(),
    side_team: document.getElementById("m_side").value.trim().toUpperCase(),
    price: Number(document.getElementById("m_price").value),
    qty: Number(document.getElementById("m_qty").value),
    fees: Number(document.getElementById("m_fees").value),
    status: document.getElementById("m_status").value,
    settlement_px: document.getElementById("m_settle").value === "" ? null : Number(document.getElementById("m_settle").value),
    notes: document.getElementById("m_notes").value.trim(),
    recorded_at: new Date().toISOString(),
    decision_day: new Date().toISOString().slice(0,10),
    units: Number(document.getElementById("m_qty").value) || 0,
    stream: "real_manual_browser",
  }
  if(!row.market_id || !row.side_team){
    alert("market_id and side team are required")
    return
  }
  if(pilotReadiness?.pause?.paused && row.status === "open"){
    alert(`Pilot paused — new open fills blocked (${(pilotReadiness.pause.reasons||[]).join(", ")}). Existing exposure is preserved.`)
    return
  }
  const existing = loadManual()
  const dup = existing.find(p =>
    p.market_id === row.market_id &&
    p.side_team === row.side_team &&
    p.status === "open" &&
    row.status === "open"
  )
  if(dup){
    alert("Duplicate open position for this market/side blocked. Settle or cancel the existing one first.")
    return
  }
  const risk = loadRisk() || {per_bet:1, daily:5, same_game:1, same_team:2}
  const check = exposureClientCheck(row, existing, risk)
  if(!check.allowed){
    alert(`Exposure limit blocked save: ${check.reasons.join(", ")}`)
    return
  }
  existing.push(row)
  localStorage.setItem(LS_MANUAL, JSON.stringify(existing))
  renderManual()
  renderPaperSummary()
}

function exposureClientCheck(proposed, existing, risk){
  const reasons = []
  const units = Number(proposed.units) || 0
  if(units > Number(risk.per_bet)) reasons.push("exceeds_per_bet_cap")
  const day = existing.filter(p => p.decision_day === proposed.decision_day && p.status !== "canceled")
    .reduce((s,p)=>s+(Number(p.units)||0), 0)
  if(day + units > Number(risk.daily)) reasons.push("exceeds_daily_cap")
  const game = existing.filter(p => String(p.game_pk) === String(proposed.game_pk) && p.status !== "canceled")
    .reduce((s,p)=>s+(Number(p.units)||0), 0)
  if(game + units > Number(risk.same_game)) reasons.push("exceeds_same_game_cap")
  const team = existing.filter(p => p.side_team === proposed.side_team && p.status !== "canceled")
    .reduce((s,p)=>s+(Number(p.units)||0), 0)
  if(team + units > Number(risk.same_team)) reasons.push("exceeds_same_team_cap")
  return {allowed: reasons.length === 0, reasons}
}

function clearManualPositions(){
  if(!confirm("Clear all manual positions from this browser?")) return
  localStorage.removeItem(LS_MANUAL)
  renderManual()
  renderPaperSummary()
}

function renderManual(){
  const rows = loadManual()
  const el = document.getElementById("manualList")
  if(!rows.length){
    el.innerHTML = `<p class="muted">No manual positions recorded in this browser.</p>`
    return
  }
  const settled = rows.filter(r => r.status === "settled")
  const open = rows.filter(r => r.status === "open")
  let settledNet = 0
  settled.forEach(r=>{
    const settle = r.settlement_px == null ? null : Number(r.settlement_px)
    if(settle == null) return
    const proceeds = settle * Number(r.qty)
    const cost = Number(r.price) * Number(r.qty) + Number(r.fees || 0)
    settledNet += proceeds - cost
  })
  const openExposure = open.reduce((s,r)=> s + Number(r.price)*Number(r.qty) + Number(r.fees||0), 0)
  let html = `<p><strong>Settled net (real/browser):</strong> ${settledNet.toFixed(4)} &nbsp;|&nbsp; <strong>Open exposure:</strong> ${openExposure.toFixed(4)} (separate from paper)</p>`
  html += `<table><tr><th>When</th><th>Market</th><th>Side</th><th>Px</th><th>Qty</th><th>Fees</th><th>Status</th><th>Settle</th><th></th></tr>`
  rows.slice().reverse().forEach(r=>{
    html += `<tr>
      <td>${(r.recorded_at||"").replace("T"," ").replace("Z","")}</td>
      <td>${r.market_id}</td><td>${r.side_team}</td>
      <td>${fmtNum(r.price)}</td><td>${fmtNum(r.qty,2)}</td><td>${fmtNum(r.fees,4)}</td>
      <td>${r.status}</td><td>${r.settlement_px == null ? "—" : fmtNum(r.settlement_px)}</td>
      <td><button class="secondary" onclick="deleteManual('${r.id}')">Delete</button></td>
    </tr>`
  })
  html += `</table>`
  el.innerHTML = html
}

function deleteManual(id){
  const next = loadManual().filter(r => r.id !== id)
  localStorage.setItem(LS_MANUAL, JSON.stringify(next))
  renderManual()
  renderPaperSummary()
}

function boardAgeSeconds(){
  const gen = boardMeta.generated_at_utc
  if(!gen) return null
  const t = Date.parse(gen)
  if(Number.isNaN(t)) return null
  return (Date.now() - t) / 1000
}

function renderReadiness(){
  const el = document.getElementById("readinessBanner")
  const dual = pilotReadiness?.dual_conclusions || {}
  const evidence = pilotReadiness?.evidence_for_limited_real_money_pilot || {}
  const pause = pilotReadiness?.pause || {}
  const verdict = evidence.verdict || boardMeta.evidence_verdict || evaluation?.verdict || "unknown"
  const status = evidence.validation_status || boardMeta.validation_status || evaluation?.validation_status || "unknown"
  const reasons = evidence.no_bet_reasons || boardMeta.gate_fail_reasons || evaluation?.gates?.gate_fail_reasons || []
  const remaining = evidence.remaining_before_pilot || []
  const limitsOk = riskConfigured()
  const softwareOk = dual.software_operates_correctly !== false
  const moneyOk = !!dual.evidence_supports_limited_real_money_pilot && limitsOk && boardMeta.betting_mode !== "disabled" && !pause.paused
  const age = boardAgeSeconds()
  const staleBoard = age != null && age > 300
  el.className = "decisionBanner " + (moneyOk ? "ok" : "warn")
  el.innerHTML = `
    <div><strong>Software readiness:</strong> ${softwareOk ? "YES — inspection / Pass / no-bet workflow operates" : "NO — incomplete artifacts"}
    (board rows=${boardMeta.n_board_rows ?? "—"}, modes shadow / betting disabled).</div>
    <div><strong>Evidence for limited real-money pilot:</strong> ${moneyOk ? "YES" : "NO — do not use real money"}.</div>
    <div><strong>Evidence verdict:</strong> ${verdict} &nbsp;|&nbsp; <strong>validation_status:</strong> ${status}
    &nbsp;|&nbsp; <strong>pilot_authorized:</strong> ${pilotReadiness?.pilot_authorized ? "yes" : "no"}</div>
    <div><strong>Pause:</strong> ${pause.paused ? `ACTIVE (${(pause.reasons||[]).join(", ")}) — existing exposure preserved` : "clear"}</div>
    <div><strong>Personal limits configured:</strong> ${limitsOk ? "yes" : "no (normalized paper units only)"}</div>
    <div><strong>Board age:</strong> ${age == null ? "—" : fmtAge(age)}${staleBoard ? " — STALE EXPORT, reload/re-export before acting" : ""}</div>
    <div class="muted" style="margin-top:8px">${(reasons || []).slice(0,8).map(r => `• ${r}`).join("<br>") || "• no gate reasons listed"}</div>
    ${remaining.length ? `<div class="muted" style="margin-top:8px"><strong>Remaining before pilot:</strong><br>${remaining.map(r=>`• ${r}`).join("<br>")}</div>` : ""}
  `
}

function renderPaperSummary(){
  const el = document.getElementById("paperSummary")
  const fixture = evaluation?.paper_ledger_fixture
  const recon = pilotReadiness?.reconcile
  const manual = loadManual()
  let html = `<h4>Paper / fixture</h4>`
  if(fixture?.hand_calculated?.example_a){
    const a = fixture.hand_calculated.example_a
    html += `<p class="muted">Hand-reconciled fixture (demo): fill ${a.filled_qty} @ notional ${a.notional}, fees ${Number(a.fees).toFixed(6)}, net-if-win ${Number(a.net_if_win).toFixed(6)}. <span class="demoTag">DEMO FIXTURE</span></p>`
  } else {
    html += `<p class="muted">No paper fixture summary loaded.</p>`
  }
  if(recon){
    html += `<h4>Reconcile (streams separate)</h4>`
    html += `<p class="muted">Real settled_net=${recon.real_manual?.settled_net ?? "—"} open=${recon.real_manual?.open_exposure ?? "—"} | Paper settled_net=${recon.paper_ledger?.settled_net ?? "—"}</p>`
    html += `<p class="muted">${recon.comparison_note || ""}</p>`
  }
  html += `<h4>Manual real positions (this browser)</h4>`
  html += `<p class="muted">${manual.length} recorded. Settled net is shown on the Manual tab, separate from open exposure and paper.</p>`
  el.innerHTML = html
}

function renderEvidence(){
  const el = document.getElementById("evidencePanel")
  const cov = evaluation?.data_coverage || {}
  el.innerHTML = `
    <h3>Registered evidence</h3>
    <div class="decisionMeta">
      <div><span>Verdict</span>${evaluation?.verdict || boardMeta.evidence_verdict || "—"}</div>
      <div><span>Validation</span>${evaluation?.validation_status || boardMeta.validation_status || "—"}</div>
      <div><span>Policy</span>${boardMeta.policy_version || "—"}</div>
      <div><span>Policy hash</span>${boardMeta.policy_hash || "—"}</div>
      <div><span>Protocol</span>${evaluation?.protocol_id || pilotReadiness?.protocol_id || "—"}</div>
      <div><span>Market</span>${pilotReadiness?.market_id || "mlb_pregame_moneyline"}</div>
      <div><span>Mapped contracts</span>${cov.n_mapped_contracts ?? "—"}</div>
      <div><span>Quote index rows</span>${cov.n_quote_index_rows ?? "—"}</div>
      <div><span>Labeled Polymarket dates</span>${cov.n_polymarket_labeled_eligible_dates ?? "—"}</div>
      <div><span>Board generated</span>${boardMeta.generated_at_utc || "—"}</div>
    </div>
    <p class="muted" style="margin-top:10px">${cov.note || ""}</p>
  `
  renderPaperSummary()
}

function clientInvalidate(row){
  const reasons = []
  if((pilotReadiness?.evidence_for_limited_real_money_pilot?.verdict || boardMeta.evidence_verdict) !== "edge_supported"){
    reasons.push("evidence_gate")
  }
  if(pilotReadiness?.pause?.paused){
    reasons.push("pilot_paused")
  }
  if(String(row.quote_stale).toLowerCase() === "true"){
    reasons.push("stale_quote")
  }
  const age = Number(row.quote_age_seconds)
  if(!Number.isNaN(age) && age > 30){
    reasons.push("quote_age_gt_30s")
  }
  const boardAge = boardAgeSeconds()
  if(boardAge != null && boardAge > 300){
    reasons.push("stale_board_export")
  }
  if(row.scheduled_start_utc){
    const start = Date.parse(row.scheduled_start_utc)
    if(!Number.isNaN(start) && Date.now() >= start){
      reasons.push("game_started")
    }
  }
  const exec = Number(row.executable_buy)
  const maxP = Number(row.max_acceptable_price)
  if(!Number.isNaN(exec) && !Number.isNaN(maxP) && exec > maxP){
    reasons.push("price_above_max")
  }
  if(!riskConfigured()){
    reasons.push("personal_limits_missing")
  }
  if(row.probability_source === "market_only_fallback" || row.probability_source === "market_mid_display_only"){
    reasons.push("market_only_not_independent")
  }
  return reasons
}

async function recheckQuote(marketSlug, marketId, sideIsLong){
  try{
    const url = `/api/recheck-quote?market_slug=${encodeURIComponent(marketSlug)}&market_id=${encodeURIComponent(marketId||"")}`
    const res = await fetch(url, {cache:"no-store"})
    const data = await res.json()
    if(!res.ok){
      alert(`Recheck failed: ${data.error || res.status}. Re-run export_polymarket_decision_board.py if the local API is not running.`)
      return
    }
    const px = sideIsLong === "True" || sideIsLong === true || sideIsLong === "true"
      ? data.best_ask
      : (data.best_bid == null ? null : 1 - Number(data.best_bid))
    alert(
      `Fresh public quote (no order placed)\n` +
      `receive=${data.receive_time_utc}\n` +
      `best_bid=${data.best_bid} best_ask=${data.best_ask}\n` +
      `side executable≈${px}\n` +
      `ask_size=${data.best_ask_size} bid_size=${data.best_bid_size}\n` +
      `eligible=${data.eligible}`
    )
  }catch(err){
    alert(`Recheck unavailable (${err.message}). Start: PYTHONPATH=src python scripts/serve_decision_dashboard.py`)
  }
}

function fillManualFromRow(row){
  document.getElementById("m_market_id").value = row.market_id || ""
  document.getElementById("m_game_pk").value = row.game_pk || ""
  document.getElementById("m_side").value = row.side_team || ""
  document.getElementById("m_price").value = row.executable_buy || ""
  document.getElementById("m_qty").value = "1"
  document.getElementById("m_fees").value = row.est_taker_fee_per_contract || ""
  selectPmTab("manual")
}

function renderBoard(){
  const filter = document.getElementById("stateFilter").value
  const el = document.getElementById("boardList")
  let rows = boardRows.slice()
  if(filter !== "all"){
    rows = rows.filter(r => r.decision_state === filter)
  }
  rows.sort((a,b)=>{
    const as = Date.parse(a.scheduled_start_utc || "") || 0
    const bs = Date.parse(b.scheduled_start_utc || "") || 0
    return as - bs
  })
  if(!rows.length){
    el.innerHTML = `<p class="muted">No decision rows. Run <code>PYTHONPATH=src python scripts/export_polymarket_decision_board.py</code>.</p>`
    return
  }
  el.innerHTML = rows.map(row=>{
    const invalidate = clientInvalidate(row)
    const demo = String(row.demo_fixture).toLowerCase() === "true"
    const stakeLabel = riskConfigured() ? "limits saved (guidance still gated)" : "normalized paper units"
    return `<article class="decisionCard">
      <div class="decisionState ${stateClass(row.decision_state)}">${row.decision_state}</div>
      ${demo ? `<div class="demoTag">DEMO FIXTURE — not production</div>` : ""}
      <h3>${row.away_team || "?"} @ ${row.home_team || "?"} — buy <em>${row.side_team}</em></h3>
      <div class="decisionMeta">
        <div><span>Contract</span><a href="${row.contract_url}" target="_blank" rel="noopener">${row.event_slug || row.market_slug || "link"}</a></div>
        <div><span>Start (UTC)</span>${row.scheduled_start_utc || "—"}</div>
        <div><span>Model p(win)</span>${fmtPct(row.model_probability)}</div>
        <div><span>Uncertainty</span>${(row.uncertainty_note || "").slice(0,120)}${(row.uncertainty_note||"").length>120?"…":""}</div>
        <div><span>Executable buy</span>${fmtNum(row.executable_buy)}</div>
        <div><span>Executable qty</span>${fmtNum(row.executable_qty,2)}</div>
        <div><span>Est. fee / contract</span>${fmtNum(row.est_taker_fee_per_contract,4)}</div>
        <div><span>Total cost / contract</span>${fmtNum(row.total_acquisition_cost_per_contract)}</div>
        <div><span>Max acceptable price</span>${fmtNum(row.max_acceptable_price)}</div>
        <div><span>Permitted exposure</span>${row.permitted_exposure_units || "1"} (${stakeLabel})</div>
        <div><span>Quote age</span>${fmtAge(row.quote_age_seconds)}</div>
        <div><span>Starters</span>${row.home_starter_status || "—"} / ${row.away_starter_status || "—"}</div>
        <div><span>Lineups</span>${row.home_lineup_status || "—"} / ${row.away_lineup_status || "—"}</div>
        <div><span>Model version</span>${row.model_version || "—"}</div>
        <div><span>Prob source</span>${row.probability_source || "—"}</div>
        <div><span>Pass reason</span>${row.pass_reason || "—"}</div>
      </div>
      <p class="muted" style="margin-top:10px">${row.uncertainty_note || ""}</p>
      <p class="muted">Live invalidate checks: ${invalidate.length ? invalidate.join(", ") : "none beyond saved state"}</p>
      <div class="btnRow">
        <button class="secondary" onclick='recheckQuote(${JSON.stringify(row.market_slug)}, ${JSON.stringify(row.market_id)}, ${JSON.stringify(row.is_long)})'>Recheck public price</button>
        <button class="secondary" onclick='fillManualFromRow(${JSON.stringify({
          market_id: row.market_id,
          game_pk: row.game_pk,
          side_team: row.side_team,
          executable_buy: row.executable_buy,
          est_taker_fee_per_contract: row.est_taker_fee_per_contract
        })})'>Copy to manual record</button>
      </div>
    </article>`
  }).join("")
}

function renderOps(){
  document.getElementById("opsGuide").textContent = `Daily operating guide — gated manual pilot (no automated orders)

1. Capture books:
   PYTHONPATH=src python scripts/capture_polymarket.py --with-baseball

2. Paper ledger cycle (optional settlement):
   PYTHONPATH=src python scripts/run_polymarket_paper_ledger.py

3. Export decision board + readiness:
   PYTHONPATH=src python scripts/export_polymarket_decision_board.py --with-live-baseball
   PYTHONPATH=src python scripts/write_pilot_readiness_report.py

4. Dashboard:
   PYTHONPATH=src python scripts/serve_decision_dashboard.py
   Open http://127.0.0.1:8765/polymarket.html

5. Read the dual readiness banner.
   - Software YES + evidence NO ⇒ correct action is no bet.
   - If pilot is paused, do not open new exposure; existing exposure remains.

6. Risk tab: enter dedicated bankroll, max affordable loss, per-bet, daily, and correlated caps.
   Until supplied, use normalized paper units only.

7. Before any manual purchase: Recheck public price, confirm max acceptable price, quote age ≤30s, board export fresh, then place yourself on Polymarket. Record the real fill under Manual positions.

8. Never increase stakes to recover losses or meet an income deadline.
   Exposure increases require the registered review protocol checkpoints.

Software works for Pass / no-bet. Evidence does NOT currently support a real-money pilot.`
}

async function reloadBoard(){
  try{
    boardRows = await loadCSV("./data/polymarket_decision_board.csv")
    boardMeta = await loadJSON("./data/polymarket_decision_meta.json")
    const prev = localStorage.getItem(LS_BOARD_SEEN)
    if(prev && boardMeta.generated_at_utc && prev !== boardMeta.generated_at_utc){
      console.log("Board regenerated since last view", prev, "->", boardMeta.generated_at_utc)
    }
    if(boardMeta.generated_at_utc){
      localStorage.setItem(LS_BOARD_SEEN, boardMeta.generated_at_utc)
    }
  }catch(err){
    boardRows = []
    boardMeta = {error: String(err)}
  }
  try{
    evaluation = await loadJSON("./data/polymarket_paper_evaluation.json")
  }catch{
    try{
      evaluation = await loadJSON("../reports/model_validation/polymarket_paper_evaluation.json")
    }catch{
      evaluation = null
    }
  }
  try{
    pilotReadiness = await loadJSON("./data/polymarket_pilot_readiness.json")
  }catch{
    try{
      pilotReadiness = await loadJSON("../reports/model_validation/polymarket_pilot_readiness.json")
    }catch{
      pilotReadiness = null
    }
  }
  let paperDefaults = null
  try{
    paperDefaults = await loadJSON("./data/polymarket_pilot_risk_limits.json")
  }catch{
    paperDefaults = pilotReadiness?.risk_limits || null
  }
  renderReadiness()
  renderBoard()
  renderEvidence()
  renderManual()
  renderOps()
  const risk = loadRisk()
  if(risk){
    document.getElementById("r_bankroll").value = risk.bankroll
    document.getElementById("r_max_loss").value = risk.max_loss
    document.getElementById("r_per_bet").value = risk.per_bet
    document.getElementById("r_daily").value = risk.daily
    document.getElementById("r_game").value = risk.same_game
    document.getElementById("r_team").value = risk.same_team
    document.getElementById("riskStatus").textContent = `Loaded browser limits saved ${risk.saved_at || ""}`
  } else if(paperDefaults){
    document.getElementById("r_bankroll").value = paperDefaults.bankroll
    document.getElementById("r_max_loss").value = paperDefaults.max_affordable_loss
    document.getElementById("r_per_bet").value = paperDefaults.per_bet_exposure_limit
    document.getElementById("r_daily").value = paperDefaults.daily_exposure_limit
    document.getElementById("r_game").value = paperDefaults.same_game_exposure_limit
    document.getElementById("r_team").value = paperDefaults.same_team_exposure_limit
    document.getElementById("riskStatus").textContent =
      `Prefilling agent paper-unit defaults (${paperDefaults.currency}). Not a personal USD bankroll. Save to use in this browser.`
  }
}

document.addEventListener("visibilitychange", ()=>{
  if(document.visibilityState === "visible"){
    renderBoard()
    renderReadiness()
  }
})

reloadBoard()
