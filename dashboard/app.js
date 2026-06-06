let trendChart;
let decisionChart;

const usd = value => `$${Number(value || 0).toFixed(2)}`;
const number = value => Number(value || 0).toLocaleString();
const el = id => document.getElementById(id);
const tokenInput = () => el("api-token");

function escapeHtml(value) {
  return String(value).replace(/[&<>"']/g, char => ({
    "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;"
  }[char]));
}
function renderSummary(summary) {
  el("cost").textContent = usd(summary.cost_usd);
  el("sessions").textContent = number(summary.sessions);
  el("agents").textContent = number(summary.agents);
  el("leaky-cost").textContent = usd(summary.leaky_cost_usd);
  el("leaky").textContent = number(summary.leaky_sessions);
  el("tokens").textContent = number(summary.tokens_billed);
}
function renderTeams(teams) {
  el("teams-body").innerHTML = teams.length ? teams.map(team => `
    <tr><td>${escapeHtml(team.team_id)}</td><td>${number(team.sessions)}</td>
    <td>${number(team.leaky_sessions)}</td><td>${usd(team.cost_usd)}</td></tr>`).join("")
    : `<tr><td colspan="4">No metrics received.</td></tr>`;
}
function renderSessions(sessions) {
  el("sessions-body").innerHTML = sessions.length ? sessions.map(session => `
    <tr><td><code>${escapeHtml(session.session_id_hash)}</code></td>
    <td>${escapeHtml(session.team_id)}</td><td>${escapeHtml(session.source)}</td>
    <td class="decision-${session.decision}">${escapeHtml(session.decision)}</td>
    <td>${(Number(session.context_pct) * 100).toFixed(0)}%</td>
    <td>${usd(session.cost_usd)}</td><td>${new Date(session.observed_at).toLocaleDateString()}</td></tr>`).join("")
    : `<tr><td colspan="7">No sessions in this period.</td></tr>`;
}
function renderCharts(data) {
  if (!window.Chart) return;
  if (trendChart) trendChart.destroy();
  if (decisionChart) decisionChart.destroy();
  const grid = { color: "#253451" };
  const ticks = { color: "#96a6c5" };
  trendChart = new Chart(el("trend-chart"), {
    type: "line",
    data: { labels: data.trend.map(row => row.day), datasets: [{
      label: "Incremental cost (USD)", data: data.trend.map(row => row.cost_usd),
      borderColor: "#35bdf5", backgroundColor: "#35bdf533", fill: true, tension: .32
    }] },
    options: { plugins: { legend: { labels: ticks } }, scales: { x: { grid, ticks }, y: { grid, ticks } } }
  });
  decisionChart = new Chart(el("decision-chart"), {
    type: "doughnut",
    data: { labels: data.decisions.map(row => row.decision), datasets: [{
      data: data.decisions.map(row => row.sessions),
      backgroundColor: ["#ffb547", "#33d29a", "#35bdf5", "#7587a8", "#d993ff"]
    }] },
    options: { plugins: { legend: { labels: ticks } } }
  });
}
async function refresh() {
  const params = new URLSearchParams({ days: el("days").value });
  if (el("team").value.trim()) params.set("team", el("team").value.trim());
  const token = tokenInput().value.trim();
  if (token) localStorage.setItem("finopsApiToken", token);
  else localStorage.removeItem("finopsApiToken");
  el("status").className = "status";
  el("status").textContent = "Loading metrics...";
  try {
    const headers = token ? { Authorization: `Bearer ${token}` } : {};
    const response = await fetch(`/aggregate?${params}`, { headers });
    if (response.status === 401) throw new Error("API token is missing or invalid");
    if (!response.ok) throw new Error(`API returned HTTP ${response.status}`);
    const data = await response.json();
    renderSummary(data.summary);
    renderTeams(data.teams);
    renderSessions(data.top_sessions);
    renderCharts(data);
    el("status").textContent = `Updated ${new Date().toLocaleTimeString()} | ${data.days}-day view`;
  } catch (error) {
    el("status").className = "status error";
    el("status").textContent = `Failed to load dashboard: ${error.message}`;
  }
}
tokenInput().value = localStorage.getItem("finopsApiToken") || "";
el("refresh").addEventListener("click", refresh);
refresh();
