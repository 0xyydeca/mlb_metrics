let wave = []

let pave = []

let confidence = []

let hitterHitPredictions = []

let playerMatches = []

let selectedTeams =
[]



async function loadCSV(path){

const response =

await fetch(

`${path}?t=${Date.now()}`,

{

cache:
"no-store"

}

)

if(!response.ok){
throw new Error(`Failed to load ${path}: ${response.status}`)
}

const text =
await response.text()

return Papa.parse(
text,
{
header:true,
skipEmptyLines:true
}
).data

}



// Quant-analytics item #5 ("backtest scope and statistical significance"):
// shared formatting for the Wilson CIs / p-values evaluation.py and
// game_evaluation.py now attach to every real backtest rate, so a bare
// percentage on the dashboard is never read as more certain than its real
// sample size supports. `low`/`high` are the *_ci_low/*_ci_high CSV
// columns (already 0-1 fractions, like the rate itself); returns "" when
// either is missing/blank rather than a broken "NaN%" string.
function ciLabel(low, high){
if(low === undefined || low === "" || high === undefined || high === ""){
return ""
}
return `95% CI ${(Number(low) * 100).toFixed(0)}–${(Number(high) * 100).toFixed(0)}%`
}

// `p` is a *_p_value CSV column (a real two-sided test p-value, see
// evaluation.binomial_significance/mean_significance). Below 0.05 is
// labeled "significant" - conventional, not a claim of certainty; above
// is labeled "not significant" explicitly rather than just omitted, since
// "no significance note shown" would otherwise read as "no evidence
// either way" instead of "checked, and it isn't distinguishable from
// noise yet." Returns "" when missing/blank (e.g. n=0, nothing to test).
function significanceLabel(p){
if(p === undefined || p === "" || Number.isNaN(Number(p))){
return ""
}
const value = Number(p)
return value < 0.05
? `significant (p=${value.toFixed(3)})`
: `not significant (p=${value.toFixed(2)})`
}



function buildTable(
data,
id,
limit=null
){

if(
limit
){

data =
data.slice(
0,
limit
)

}

if(
!data.length
){

document
.getElementById(
id
).innerHTML =
"No data"

return

}

// Quant-analytics item #3, slice 3 ("confidence intervals"): fold any
// "<X>_CI_Low"/"<X>_CI_High" pair - alongside a matching bare "<X>"
// column - into ONE formatted cell on X ("28.4% (22.1%-35.2%)"),
// dropping the two raw CI columns from the table entirely, rather than
// showing three separate raw, unformatted numbers. Data-driven (keys
// present on data[0]), so this is a no-op for any table with no CI
// columns (e.g. Top PAVE Players) - and forward-compatible: any future
// column following this same naming convention gets folded in for free.
const allKeys =
Object.keys(
data[0]
)
const ciBases =
allKeys
.filter(
k=>k.endsWith("_CI_Low")
)
.map(
k=>k.slice(0, -"_CI_Low".length)
)
.filter(
base=>allKeys.includes(base) && allKeys.includes(base + "_CI_High")
)
const hiddenKeys =
new Set(
ciBases.flatMap(base=>[base + "_CI_Low", base + "_CI_High"])
)
const columns =
allKeys.filter(k=>!hiddenKeys.has(k))
const pct =
v=>(Number(v) * 100).toFixed(1) + "%"

let html =
"<table>"

html +=
"<tr>"

columns
.forEach(
c=>
html +=
`<th>${c}</th>`
)

html +=
"</tr>"

data.forEach(
row=>{

html +=
"<tr>"

columns
.forEach(
c=>{
if(ciBases.includes(c)){
html += `<td>${pct(row[c])} (${pct(row[c + "_CI_Low"])}-${pct(row[c + "_CI_High"])})</td>`
} else {
html += `<td>${row[c]}</td>`
}
}
)

html +=
"</tr>"

}
)

html +=
"</table>"

document
.getElementById(
id
)
.innerHTML =
html

}



function searchPlayer(){

const query =

document
.getElementById(
"playerSearch"
)
.value
.trim()
.toLowerCase()

const type =

document
.getElementById(
"playerType"
)
.value

const source =

type==="wave"
?
wave
:
pave

playerMatches =

source
.filter(
p=>{

const full =

`${p.name_first}
${p.name_last}`

.toLowerCase()

return full.includes(
query
)

}
)

.slice(
0,
10
)

if(
playerMatches.length===0
){

document
.getElementById(
"playerChoices"
)
.innerHTML =
"No players found"

document
.getElementById(
"playerResult"
)
.innerHTML =
""

return

}

let html =
""

playerMatches
.forEach(
(
p,
i
)=>{

html +=

`

<button

onclick="showPlayer(
${i}
)"

>

${p.name_first}
${p.name_last}

</button>

`

}

)

document
.getElementById(
"playerChoices"
)
.innerHTML =
html

}



function showPlayer(index){

const player =

playerMatches[
index
]

const playerId =

player.key_mlbam

const modelPrediction =

hitterHitPredictions.find(
p=>
p.key_mlbam
===
playerId
)

const isPave =

document
.getElementById(
"playerType"
)
.value

===

"pave"

document
.getElementById(
"playerResult"
)
.innerHTML =

`

<div
class="infoCard">

<div
class="cardHeader">

<div
style="
display:flex;
gap:16px;
align-items:center;
">

<img

src="${playerImage(playerId)}"

class="playerHeadshot"

alt="${player.name_first} ${player.name_last}"

>

<div>

<div
class="cardTitle">

${player.name_first}

${player.name_last}

</div>

<div
style="
opacity:.7;
margin-top:4px;
display:flex;
align-items:center;
gap:8px;
">

<img

src=
"${teamLogo(
player.team
)}"

class=
"smallTeamLogo"

onerror=
"this.style.display='none'"

>

${player.team}

•

${
isPave
?
"PAVE"
:
"WAVE"
}

</div>

</div>

</div>

</div>

<div
class="cardStats">

<div>

Team

</div>

<div>

${player.team || "-"}

</div>

${
isPave

?

`

<div>

Expected Hits

</div>

<div>

${Number(
player.Expected_Hits
||
0
)
.toFixed(
2
)}

</div>

<div>

Expected Bases

</div>

<div>

${Number(
player.Expected_Bases
||
0
)
.toFixed(
2
)}

</div>

<div>

Expected HR

</div>

<div>

${Number(
player.Expected_HRs
||
0
)
.toFixed(
2
)}

</div>

<div>

At Bats

</div>

<div>

${player.at_bats}

</div>

`

:

`

<div>

Heuristic

</div>

<div>

${(
Number(
player.Approach
||
0
)
*
100
)
.toFixed(
1
)}%

</div>

<div>

Model Odds

</div>

<div>

${
modelPrediction
?
(
Number(
modelPrediction.Model_Hit_Probability
||
0
)
*
100
)
.toFixed(
1
)+"%"
:
"-"
}

</div>

<div>

Expected Bases

</div>

<div>

${Number(
player.Expected_Bases
||
0
)
.toFixed(
2
)}

</div>

<div>

PA

</div>

<div>

${
Number(
player.PA_L
||
0
)

+

Number(
player.PA_R
||
0
)

}

</div>

`

}

</div>

</div>

`

}



function loadTeamExplorer(){

const container =

document
.getElementById(
"teamButtons"
)

container.innerHTML =
""

const values =

confidence
.map(
r=>

Number(
r.Confidence
||
0
)

)

const min =

Math.min(
...values
)

const max =

Math.max(
...values
)

container.style.display =
"flex"

container.style.flexWrap =
"wrap"

container.style.gap =
"6px"

confidence.forEach(
team=>{

const val =

Number(
team.Confidence
)

const pct =

(
val
-
min
)

/

(
max
-
min
)

const red =

Math.round(
255
*
pct
)

const blue =

Math.round(
255
*
(
1
-
pct
)
)

const btn =

document.createElement(
"button"
)

btn.style.width =
"60px"

btn.style.height =
"44px"

btn.style.position =
"relative"

btn.style.border =
"none"

btn.style.borderRadius =
"8px"

btn.style.background =
"#222"

btn.style.color =
"white"

btn.style.cursor =
"pointer"

btn.innerHTML =

`

<div
style="
font-weight:bold;
">

${team.team}

</div>

<div
style="
position:absolute;
left:0;
bottom:0;
height:6px;
width:100%;
border-radius:0 0 8px 8px;

background:

linear-gradient(
90deg,

rgb(
${red},
80,
${blue}
),

rgb(
${red},
80,
${blue}
)

);

">

</div>

`

btn.onclick =
()=>{

selectTeam(
team.team
)

}

container
.appendChild(
btn
)

}

)

}

function selectTeam(team){

selectedTeams =

selectedTeams.filter(
t=>
t!==team
)

selectedTeams.push(
team
)

if(
selectedTeams.length
>
2
){

selectedTeams.shift()

}

showTeam()

}

function playerImage(id){

if(
!id
){

return ""

}

return `https://img.mlbstatic.com/mlb-photos/image/upload/w_240,q_auto:best/v1/people/${String(id).trim()}/headshot/67/current`

}

function teamLogo(team){

const ids = {

AZ:109,
ATL:144,
BAL:110,
BOS:111,
CHC:112,
CWS:145,
CIN:113,
CLE:114,
COL:115,
DET:116,
HOU:117,
KC:118,
LAA:108,
LAD:119,
MIA:146,
MIL:158,
MIN:142,
NYM:121,
NYY:147,
ATH:133,
PHI:143,
PIT:134,
SD:135,
SF:137,
SEA:136,
STL:138,
TB:139,
TEX:140,
TOR:141,
WSH:120

}

const key =

String(
team
)

.trim()

.toUpperCase()

return ids[key]

?

`https://www.mlbstatic.com/team-logos/team-cap-on-light/${ids[key]}.svg`

:

""

}

function showTeam(){

const results =

selectedTeams

.map(
team=>

confidence.find(
r=>

r.team===team

)

)

.filter(
Boolean
)

if(
results.length===0
){

document
.getElementById(
"teamResult"
)
.innerHTML =
""

return

}

const values =

confidence.map(
r=>

Number(
r.Confidence
)

)

const min =

Math.min(
...values
)

const max =

Math.max(
...values
)

const winning = [

"current",

"Strength",

"pyth_Strength",

"SOS",

"pyth_SOS",

"Confidence",

"pyth_Confidence"

]

const offense = [

"true_power",

"offensive_edge",

"suppression_resistance",

"home_run_reliance",

"homer_per_game",

"game_homer_rate"

]

const park = [

"team_home_run_rate",

"away_hr_rate",

"Park_Factor"

]

function rows(
result,
list
){

const other =

results.find(
r=>

r.team
!==

result.team

)

return list

.map(
k=>{

const value =

Number(
result[k]
||
0
)

let arrow =
""

if(
results.length===2
&&
other
){

const compare =

Number(
other[k]
||
0
)

if(
value
>
compare
){

arrow =

'<span class="compareUp">▲</span>'

}

else if(
value
<
compare
){

arrow =

'<span class="compareDown">▼</span>'

}

}

return `

<div>

${k.replaceAll(
"_",
" "
)}

</div>

<div>

${value.toFixed(
3
)}

${arrow}

</div>

`

})

.join(
""

)

}

document
.getElementById(
"teamResult"
)
.innerHTML =

results

.map(
result=>{

const pct =

(

Number(
result.Confidence
)

-

min

)

/

(

max

-

min

)

const color =

`rgb(

${Math.round(
255*pct
)},

80,

${Math.round(
255*(1-pct)
)}

)`

return `

<div
class="infoCard">

<div
class="cardHeader">

<div
style="
display:flex;
align-items:center;
gap:12px;
">

<img

src=
"${teamLogo(
result.team
)}"

class=
"teamLogo"

onerror=
"this.style.display='none'"

>

<div
class="cardTitle">

${result.team}

</div>

</div>

<div
class="cardBadge"

style="
background:
${color}
"

>

</div>

</div>

<div
class="profileColumns">

<div
class="profileColumn">

<div
class="profileHeader">

Winning Profile

</div>

<div
class="profileRows">

${rows(
result,
winning
)}

</div>

</div>

<div
class="profileColumn">

<div
class="profileHeader">

Offensive Profile

</div>

<div
class="profileRows">

${rows(
result,
offense
)}

</div>

</div>

<div
class="profileColumn">

<div
class="profileHeader">

Ballpark Profile

</div>

<div
class="profileRows">

${rows(
result,
park
)}

</div>

</div>

</div>

</div>

`

}

)

.join(
""

)

}

async function loadAll(){

wave =
await loadCSV(
"./data/wave.csv"
)

pave =
await loadCSV(
"./data/pave.csv"
)

confidence =
await loadCSV(
"./data/confidence.csv"
)

confidence =

confidence

.sort(
(
a,
b
)=>

String(
a.team
)
.localeCompare(

String(
b.team
)

)

)

const meanPA =

wave
.reduce(
(
sum,
p
)=>

sum

+

(

Number(
p.PA_L
||
0
)

+

Number(
p.PA_R
||
0
)

),

0

)

/

wave.length

const filteredWave =

wave
.filter(
p=>

(

Number(
p.PA_L
||
0
)

+

Number(
p.PA_R
||
0
)

)

>=

meanPA

)

buildTable(

filteredWave,

"waveTable",

20

)

const maxAB =

Math.max(

...

pave.map(
p=>

Number(
p.at_bats
||
0
)

)

)

const minAB =

maxAB
*
0.25

const filteredPave =

pave
.filter(
p=>

Number(
p.at_bats
||
0
)

>=

minAB

)

buildTable(

filteredPave,

"paveTable",

20

)

loadTeamExplorer()

}

function selectBtsView(view){
document.querySelectorAll("#btsTabs .tabButton").forEach(btn=>{
btn.classList.toggle("active", btn.dataset.view === view)
})
document.getElementById("ourPicksSection").style.display = view === "ourPicks" ? "" : "none"
document.getElementById("hitStreaksSection").style.display = view === "hitStreaks" ? "" : "none"
document.getElementById("modelOddsSection").style.display = view === "modelOdds" ? "" : "none"
}

async function loadBeatTheStreak(){

let summary = []
let picks = []

try{
summary = await loadCSV("./data/beat_the_streak_summary.csv")
}catch(e){
console.log("no beat_the_streak_summary.csv yet", e)
}

try{
picks = await loadCSV("./data/beat_the_streak_picks.csv")
}catch(e){
console.log("no beat_the_streak_picks.csv yet", e)
}

renderStreakStats(summary)
renderTodaysPicks(picks)
renderStreakHistory(picks)

}

function renderStreakStats(summary){

const el = document.getElementById("streakStats")

if(!summary.length){
el.innerHTML = "No picks tracked yet"
return
}

const s = summary[0]

const survivalRate =
s.day_survival_rate && s.day_survival_rate !== ""
? (Number(s.day_survival_rate) * 100).toFixed(1) + "%"
: "-"

// Quant-analytics item #5: the real CI on day_survival_rate, so a small
// n_days_resolved reads as "wide uncertainty band," not false precision.
const survivalSub = s.n_days_resolved > 0 ? ciLabel(s.day_survival_rate_ci_low, s.day_survival_rate_ci_high) : ""

const stat = (value, label, sub) =>
`<div class="streakStat"><div class="value">${value}</div><div class="label">${label}</div>${sub ? `<div class="sub">${sub}</div>` : ""}</div>`

el.innerHTML =
stat(s.current_streak || 0, "Current Streak") +
stat(s.longest_streak || 0, "Longest Streak") +
stat(survivalRate, "Day Survival Rate", survivalSub) +
stat(s.n_days_resolved || 0, "Days Tracked")

}

function renderTodaysPicks(picks){

const el = document.getElementById("todaysPicks")

// A date with ZERO logged candidates at all (a genuine off day - an
// All-Star break, a rainout across the whole slate, or a pipeline gap)
// still gets a "no_pick" row (see evaluation.build_beat_the_streak_export)
// so the latest date here is always the actual most recent day the
// pipeline ran. A weak-slate day no longer gets this treatment - it shows
// its real top candidates graded "speculative" instead (see below).
if(!picks.length){
el.innerHTML = "No picks tracked yet"
return
}

const latestDate = picks
.map(p=>p.date)
.sort()
.slice(-1)[0]

const todays = picks
.filter(p=>p.date === latestDate && p.status !== "no_pick")
.sort((a,b)=>Number(a.rank) - Number(b.rank))

if(!todays.length){
el.innerHTML = `<div class="pickCard no_pick"><div class="pickName">No games or eligible batters for ${latestDate}</div></div>`
return
}

const statusLabels = {
hit: "hit",
miss: "miss",
no_game: "no game",
pending: "pending",
}

const gradeLabels = {
recommended: "Recommended",
speculative: "Speculative",
}

el.innerHTML = todays
.map(p=>{

const prob =
p.predicted_probability && p.predicted_probability !== ""
? (Number(p.predicted_probability) * 100).toFixed(1) + "% predicted"
: ""

// "recommended" reuses the same highlight ring the Automated Game
// Picks cards already use for a confident pick (see .pickCard.recommended
// in style.css); "speculative" is a real, already-qualified candidate
// that just didn't clear the recommendation bar - shown, not hidden,
// but visually dimmed and explicitly labeled rather than blended in
// as if it were equally confident.
return `
<div class="pickCard ${p.status} ${p.grade || ""}">
<div class="pickName">${p.name}</div>
<div class="pickGrade ${p.grade || ""}">${gradeLabels[p.grade] || ""}</div>
<div class="pickProb">${prob}</div>
<div class="pickStatus">${statusLabels[p.status] || p.status}</div>
</div>
`

})
.join("")

}

function renderStreakHistory(picks){

const sorted = picks
.slice()
.sort((a,b)=>b.date.localeCompare(a.date) || Number(a.rank) - Number(b.rank))

buildTable(
sorted,
"streakHistoryTable",
100
)

}

async function loadHitStreaks(){

let streaks = []

try{
streaks = await loadCSV("./data/hit_streaks.csv")
}catch(e){
console.log("no hit_streaks.csv yet", e)
}

renderHitStreaks(streaks)

}

function renderHitStreaks(streaks){

if(!streaks.length){
document.getElementById("hitStreaksTable").innerHTML = "No hit streak data yet"
return
}

const top = streaks
.slice()
.sort((a,b)=>Number(b.Current_Hit_Streak) - Number(a.Current_Hit_Streak))
.slice(0, 10)
.map(p=>({
"Player": `${p.name_first} ${p.name_last}`,
"Team": p.team,
"Current Streak": p.Current_Hit_Streak,
}))

buildTable(top, "hitStreaksTable")

}

async function loadModelOdds(){

try{
hitterHitPredictions = await loadCSV("./data/hitter_hit_predictions.csv")
}catch(e){
console.log("no hitter_hit_predictions.csv yet", e)
}

renderModelOdds(hitterHitPredictions)

}

function renderModelOdds(predictions){

if(!predictions.length){
document.getElementById("modelOddsTable").innerHTML = "No model odds published yet"
return
}

const top = predictions
.slice()
.sort((a,b)=>Number(b.Model_Hit_Probability) - Number(a.Model_Hit_Probability))
.slice(0, 10)
.map(p=>({
"Player": `${p.name_first} ${p.name_last}`,
"Team": p.team,
"Opp": `${p.is_home === "True" || p.is_home === "true" ? "vs" : "@"} ${p.opponent}`,
"Model Odds": `${(Number(p.Model_Hit_Probability) * 100).toFixed(1)}%`,
}))

buildTable(top, "modelOddsTable")

}

async function loadGamePicks(){

let summary = []
let picks = []

try{
summary = await loadCSV("./data/game_picks_summary.csv")
}catch(e){
console.log("no game_picks_summary.csv yet", e)
}

try{
picks = await loadCSV("./data/game_picks_picks.csv")
}catch(e){
console.log("no game_picks_picks.csv yet", e)
}

renderGamePickStats(summary)
renderTodaysGamePicks(picks)
renderGamePickHistory(picks)

}

function renderGamePickStats(summary){

const el = document.getElementById("gamePickStats")

if(!summary.length){
el.innerHTML = "No picks tracked yet"
return
}

const s = summary[0]

const stat = (value, label, sub) =>
`<div class="streakStat"><div class="value">${value}</div><div class="label">${label}</div>${sub ? `<div class="sub">${sub}</div>` : ""}</div>`

// Quant-analytics item #6 follow-up: tracking is scoped to whether a real
// bet was ADVISED (a real Kelly edge actually cleared - see
// game_predictions.advise_bets), not the model's own above_threshold
// confidence gate, and the headline number is real money won/lost, not
// pick accuracy. "-" (same convention as beatClosingLine below) until
// n_bets_advised clears 0.
const hasBets = s.n_bets_advised && Number(s.n_bets_advised) > 0

const winRate = hasBets
? (Number(s.win_rate_on_advised_bets) * 100).toFixed(1) + "%"
: "-"

// Quant-analytics item #5: win rate's real CI - informational only, no
// significance flag here (a win/loss count alone can't tell a good -150
// favorite bet apart from a bad one the real profit already prices in -
// see evaluation.mean_significance's own docstring). roi_p_value below is
// the correctly-posed test for whether the P&L is real.
const winRateSub = hasBets ? ciLabel(s.win_rate_on_advised_bets_ci_low, s.win_rate_on_advised_bets_ci_high) : ""

// Units, not dollars - the standard bankroll-agnostic sports-betting
// convention (config.UNIT_SIZE_FRACTION of bankroll per unit). "u" suffix
// matches how bettors actually write this ("+12.4u").
const pnl = hasBets
? (Number(s.total_profit_units) >= 0 ? "+" : "") + Number(s.total_profit_units).toFixed(2) + "u"
: "-"

const pnlColor = hasBets
? (Number(s.total_profit_units) >= 0 ? "var(--success)" : "var(--danger)")
: "inherit"

// Quant-analytics item #5: the real test for "is this P&L distinguishable
// from breaking even" - a one-sample t-test on each advised bet's real
// profit (evaluation.mean_significance), not a win-rate coin-flip test.
const pnlSub = hasBets ? significanceLabel(s.roi_p_value) : ""

// Primary market skill: paired Brier difference vs true closing line
// (model − market; negative is better). beat_closing_line_rate remains
// secondary (win-count style) and does not replace magnitude-aware scoring.
const hasPairedMarket = s.n_paired_market_compared > 0 && s.model_minus_market_brier !== "" && s.model_minus_market_brier != null
const pairedBrier = hasPairedMarket
  ? Number(s.model_minus_market_brier).toFixed(4)
  : "-"
const pairedBrierSub = hasPairedMarket
  ? [
      ciLabel(s.model_minus_market_brier_ci_low, s.model_minus_market_brier_ci_high),
      s.n_paired_market_compared ? `n=${s.n_paired_market_compared}` : "",
    ].filter(Boolean).join(" · ")
  : ""

// Secondary: % of games with lower squared error vs closing line.
const hasClosingLineData = s.beat_closing_line_rate && s.n_beat_closing_line_compared > 0
const beatClosingLine = hasClosingLineData
  ? (Number(s.beat_closing_line_rate) * 100).toFixed(1) + "%"
  : "-"
const beatClosingLineSub = hasClosingLineData
  ? [ciLabel(s.beat_closing_line_rate_ci_low, s.beat_closing_line_rate_ci_high), significanceLabel(s.beat_closing_line_rate_p_value)]
      .filter(Boolean).join(" · ")
  : ""

// Streak is DAYS, not bets (2026-08-25 - "the streak should be days...
// if the cumulative bets made money that day" - see
// game_evaluation._bet_pnl_metrics' own docstring): a day with several
// advised bets counts for at most one real streak step either way,
// scored on that day's TOTAL profit, not any single bet in isolation.
el.innerHTML =
stat(s.current_bet_streak || 0, "Day Streak") +
stat(s.best_bet_streak || 0, "Best Day Streak") +
stat(winRate, "Win Rate", winRateSub) +
stat(s.n_bets_advised || 0, "Bets Tracked") +
`<div class="streakStat"><div class="value" style="color:${pnlColor}">${pnl}</div><div class="label">P&amp;L</div>${pnlSub ? `<div class="sub">${pnlSub}</div>` : ""}</div>` +
stat(pairedBrier, "Δ Brier vs Close", pairedBrierSub) +
stat(beatClosingLine, "% Lower SE", beatClosingLineSub)

}

function renderTodaysGamePicks(picks){

const el = document.getElementById("todaysGamePicks")

// Every scheduled game is published now, not just ones a real bet was
// advised on (see game_predictions.advise_bets) - advised games are
// highlighted via the "recommended" class below instead of being the
// only ones shown.
if(!picks.length){
el.innerHTML = "No game picks published yet"
return
}

const latestDate = picks
.map(p=>p.date)
.sort()
.slice(-1)[0]

const todays = picks
.filter(p=>p.date === latestDate)
.sort((a,b)=>Number(b.predicted_probability) - Number(a.predicted_probability))

if(!todays.length){
el.innerHTML = "No games today"
return
}

const statusLabels = {
win: "win",
loss: "loss",
not_played: "not played",
pending: "pending",
}

el.innerHTML = todays
.map(p=>{

const prob =
p.predicted_probability && p.predicted_probability !== ""
? (Number(p.predicted_probability) * 100).toFixed(1) + "% predicted"
: ""

// bet_units is the real "was a bet advised" signal - 0 (or blank/NaN on
// an older row) means no bet, any positive number is real units to risk.
const betUnits = Number(p.bet_units)
const betAdvised = betUnits > 0

const betLine = betAdvised
? `<div class="pickStatus">Bet advised: ${p.bet_team} @ ${p.bet_moneyline} (${betUnits.toFixed(2)}u)</div>`
: ""

return `
<div class="pickCard ${p.status}${betAdvised ? " recommended" : ""}">
<div class="pickName">${p.predicted_winner} (${p.away_team} @ ${p.home_team})</div>
<div class="pickProb">${prob}</div>
<div class="pickStatus">${statusLabels[p.status] || p.status}</div>
${betLine}
</div>
`

})
.join("")

}

function renderGamePickHistory(picks){

const sorted = picks
.slice()
.sort((a,b)=>b.date.localeCompare(a.date) || Number(b.predicted_probability) - Number(a.predicted_probability))

const statusLabels = {
win: "win",
loss: "loss",
not_played: "not played",
pending: "pending",
}

const pct = v =>
v !== undefined && v !== null && v !== "" && !Number.isNaN(Number(v))
? (Number(v) * 100).toFixed(1) + "%"
: "-"

// Real follow-up (2026-08-28 - "we're dumping almost everything data
// wise into the table, and it all gets jumbled"): trimmed to a curated
// set of columns, nicely labeled/formatted - not a raw dump of
// whatever's in game_picks_picks.csv. renderTodaysGamePicks (above)
// still reads the full row for its own cards, so nothing is removed
// from the underlying data, just from THIS table's display.
const formatted = sorted.map(p=>({
"Date": p.date,
"Predicted Winner": p.predicted_winner,
"Predicted Loser": p.predicted_loser,
"Model Probability": pct(p.predicted_probability),
"Market Probability": pct(p.market_predicted_winner_probability),
"Bet Units": p.bet_units && Number(p.bet_units) > 0 ? Number(p.bet_units).toFixed(2) : "-",
"Bet Team": p.bet_team || "-",
"Status": statusLabels[p.status] || p.status,
"Bet Profit Units":
p.bet_profit_units !== undefined && p.bet_profit_units !== "" && !Number.isNaN(Number(p.bet_profit_units))
? (Number(p.bet_profit_units) >= 0 ? "+" : "") + Number(p.bet_profit_units).toFixed(2) + "u"
: "-",
}))

buildTable(
formatted,
"gamePickHistoryTable",
100
)

}

async function loadProbablePitchers(){

let rows = []

try{
rows = await loadCSV("./data/probable_pitchers.csv")
}catch(e){
console.log("no probable_pitchers.csv yet", e)
}

const el = document.getElementById("probablePitchersTable")

if(!rows.length){
el.innerHTML = "No probable pitchers yet"
return
}

const formatted = rows.map(r=>({
"Team": r.team,
"Opponent": r.opponent,
"Home/Away": r.is_home === "True" || r.is_home === "true" ? "Home" : "Away",
"Probable Pitcher": r.pitcher_name || "TBD",
"Throws": r.Throws || "-",
"PAVE": r.PAVE && r.PAVE !== "" ? Number(r.PAVE).toFixed(3) : "-",
"PAVE+": r.PAVE_PLUS && r.PAVE_PLUS !== "" ? Number(r.PAVE_PLUS).toFixed(2) : "-",
"Power A+": r.Power_A_PLUS && r.Power_A_PLUS !== "" ? Number(r.Power_A_PLUS).toFixed(2) : "-",
}))

buildTable(
formatted,
"probablePitchersTable"
)

}

loadAll()

loadBeatTheStreak()

loadHitStreaks()

loadModelOdds()

loadGamePicks()

loadProbablePitchers()
