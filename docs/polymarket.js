/* Polymarket manual decision dashboard — extends MLB Metrics docs app.
 * No automated orders. Personal limits/positions stay in localStorage.
 */
const LS_RISK = "mlb_metrics_pm_risk_v1"
const LS_MANUAL = "mlb_metrics_pm_manual_v1"
const LS_BOARD_SEEN = "mlb_metrics_pm_board_generated_at"

let boardRows = []
let boardMeta = {}
let evaluation = null

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
  return "state-pass"
}

function riskConfigured(){
  const r = loadRisk()
  return r && Number(r.bankroll) > 0 && Number(r.max_loss) > 0
}

function loadRisk(){
  try{ return JSON.parse(localStorage.getItem(LS_RISK) || "null") }catch{ return null }
}

function saveRiskLimits(){
  const payload = {
    bankroll: Number(document.getElementById("r_bankroll").value),
    max_loss: Number(document.getElementById("r_max_loss").value),
    per_bet: Number(document.getElementById("r_per_bet").value),
    daily: Number(document.getElementById("r_daily").value),
    same_game: Number(document.getElementById("r_game").value),
    same_team: Number(document.getElementById("r_team").value),
    saved_at: new Date().toISOString(),
  }
  if(!(payload.bankroll > 0) || !(payload.max_loss > 0)){
    document.getElementById("riskStatus").textContent =
      "Bankroll and max affordable loss are required before stake guidance can enable."
    renderReadiness()
    return
  }
  if(payload.max_loss > payload.bankroll){
    document.getElementById("riskStatus").textContent =
      "Max affordable loss cannot exceed bankroll."
    return
  }
  localStorage.setItem(LS_RISK, JSON.stringify(payload))
  document.getElementById("riskStatus").textContent =
    `Saved. Stake guidance still requires a passing evidence gate (currently: ${boardMeta.evidence_verdict || "unknown"}). Stakes never auto-increase after losses.`
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
  }
  if(!row.market_id || !row.side_team){
    alert("market_id and side team are required")
    return
  }
  const existing = loadManual()
  // Duplicate guard: same market+side+open status
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
    el.innerHTML = `<p class="muted">No manual positions recorded.</p>`
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
  let html = `<p><strong>Settled net:</strong> ${settledNet.toFixed(4)} &nbsp;|&nbsp; <strong>Open exposure:</strong> ${openExposure.toFixed(4)} (separate)</p>`
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

function renderReadiness(){
  const el = document.getElementById("readinessBanner")
  const verdict = boardMeta.evidence_verdict || evaluation?.verdict || "unknown"
  const status = boardMeta.validation_status || evaluation?.validation_status || "unknown"
  const reasons = boardMeta.gate_fail_reasons || evaluation?.gates?.gate_fail_reasons || []
  const limitsOk = riskConfigured()
  const softwareOk = true
  const moneyOk = verdict === "edge_supported" && status === "validated_passed" && limitsOk && boardMeta.betting_mode !== "disabled"
  el.className = "decisionBanner " + (moneyOk ? "ok" : "warn")
  el.innerHTML = `
    <div><strong>Software readiness:</strong> works for inspection and Pass / no-bet decisions
    (board rows=${boardMeta.n_board_rows ?? "—"}, modes shadow / betting disabled).</div>
    <div><strong>Real-money readiness:</strong> ${moneyOk ? "NOT ENABLED — gates would still need live authorization" : "NO — do not use real money from this board"}.</div>
    <div><strong>Evidence verdict:</strong> ${verdict} &nbsp;|&nbsp; <strong>validation_status:</strong> ${status}</div>
    <div><strong>Personal limits configured:</strong> ${limitsOk ? "yes" : "no (stake guidance suppressed)"}</div>
    <div class="muted">${(reasons || []).slice(0,6).map(r => `• ${r}`).join("<br>") || "• no gate reasons listed"}</div>
  `
}

function renderPaperSummary(){
  const el = document.getElementById("paperSummary")
  const fixture = evaluation?.paper_ledger_fixture
  const manual = loadManual()
  let html = `<h4>Paper / fixture</h4>`
  if(fixture?.hand_calculated?.example_a){
    const a = fixture.hand_calculated.example_a
    html += `<p class="muted">Hand-reconciled fixture (demo): fill ${a.filled_qty} @ notional ${a.notional}, fees ${Number(a.fees).toFixed(6)}, net-if-win ${Number(a.net_if_win).toFixed(6)}. <span class="demoTag">DEMO FIXTURE</span></p>`
  } else {
    html += `<p class="muted">No paper fixture summary loaded.</p>`
  }
  html += `<h4>Manual real positions (this browser)</h4>`
  html += `<p class="muted">${manual.length} recorded. Settled net is shown on the Manual tab, separate from open exposure.</p>`
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
  // Stale tab / expiry checks in the browser without inventing new scores.
  const reasons = []
  if(boardMeta.evidence_verdict !== "edge_supported"){
    reasons.push("evidence_gate")
  }
  if(String(row.quote_stale).toLowerCase() === "true"){
    reasons.push("stale_quote")
  }
  const age = Number(row.quote_age_seconds)
  if(!Number.isNaN(age) && age > 30){
    reasons.push("quote_age_gt_30s")
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
  // Prefer upcoming / non-started, then by start time
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
    return `<article class="decisionCard">
      <div class="decisionState ${stateClass(row.decision_state)}">${row.decision_state}</div>
      ${demo ? `<div class="demoTag">DEMO FIXTURE — not production</div>` : ""}
      <h3>${row.away_team || "?"} @ ${row.home_team || "?"} — buy <em>${row.side_team}</em></h3>
      <div class="decisionMeta">
        <div><span>Contract</span><a href="${row.contract_url}" target="_blank" rel="noopener">${row.event_slug || row.market_slug || "link"}</a></div>
        <div><span>Start (UTC)</span>${row.scheduled_start_utc || "—"}</div>
        <div><span>Model p(win)</span>${fmtPct(row.model_probability)}</div>
        <div><span>Market mid</span>${fmtPct(row.market_mid_probability)}</div>
        <div><span>Executable buy</span>${fmtNum(row.executable_buy)}</div>
        <div><span>Executable qty</span>${fmtNum(row.executable_qty,2)}</div>
        <div><span>Est. fee / contract</span>${fmtNum(row.est_taker_fee_per_contract,4)}</div>
        <div><span>Total cost / contract</span>${fmtNum(row.total_acquisition_cost_per_contract)}</div>
        <div><span>Expected net / contract</span>${fmtNum(row.expected_net_value_per_contract)}</div>
        <div><span>Max acceptable price</span>${fmtNum(row.max_acceptable_price)}</div>
        <div><span>Quote age</span>${fmtAge(row.quote_age_seconds)}</div>
        <div><span>Starters</span>${row.home_starter_status || "—"} / ${row.away_starter_status || "—"}</div>
        <div><span>Lineups</span>${row.home_lineup_status || "—"} / ${row.away_lineup_status || "—"}</div>
        <div><span>Model version</span>${row.model_version || "—"}</div>
        <div><span>Prob source</span>${row.probability_source || "—"}</div>
        <div><span>Paper exposure units</span>${row.permitted_exposure_units || "1"} (normalized)</div>
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
  document.getElementById("opsGuide").textContent = `Daily operating guide (manual only)

1. Capture books (optional but recommended):
   PYTHONPATH=src python scripts/capture_polymarket.py --with-baseball

2. Export the decision board:
   PYTHONPATH=src python scripts/export_polymarket_decision_board.py --with-live-baseball

3. Start the local dashboard (enables quote recheck API):
   PYTHONPATH=src python scripts/serve_decision_dashboard.py
   Open http://127.0.0.1:8765/polymarket.html

4. Read the readiness banner. If evidence_verdict is insufficient_evidence, the correct action is no bet.

5. Configure personal bankroll + max affordable loss under Risk limits before any stake guidance can appear.

6. For any Paper candidate: Recheck public price, confirm max acceptable price, then place manually on Polymarket if you choose. Record the real fill under Manual positions.

7. Never increase stakes to recover losses. Never treat a stale browser tab as live.

Software works for Pass / no-bet visibility. Prospective evidence does NOT currently support using real money.`
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
  // Prefer copying evaluation into docs/data on export — fallback already in meta
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
    document.getElementById("riskStatus").textContent = `Loaded limits saved ${risk.saved_at || ""}`
  }
}

// Stale-tab: re-check ages when tab becomes visible again
document.addEventListener("visibilitychange", ()=>{
  if(document.visibilityState === "visible"){
    renderBoard()
    renderReadiness()
  }
})

reloadBoard()
