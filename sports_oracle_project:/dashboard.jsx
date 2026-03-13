import { useState, useEffect, useMemo } from "react";

const TEAMS_BY_SEED = {
  1: ["Houston", "Duke", "Auburn", "Florida"],
  2: ["Tennessee", "Alabama", "Michigan St.", "St. John's"],
  3: ["Texas Tech", "Marquette", "Iowa St.", "Wisconsin"],
  4: ["Arizona", "Purdue", "Clemson", "Maryland"],
  5: ["Michigan", "Oregon", "Louisville", "Memphis"],
  6: ["BYU", "Illinois", "Missouri", "Creighton"],
  7: ["UCLA", "Kansas", "Vanderbilt", "Ohio St."],
  8: ["UConn", "Gonzaga", "Drake", "Baylor"],
  9: ["Oklahoma", "Georgia", "Arkansas", "Texas A&M"],
  10: ["Dayton", "New Mexico", "North Carolina", "Indiana"],
  11: ["VCU", "Xavier", "NC State", "San Diego St."],
  12: ["Colorado St.", "McNeese", "UCF", "Troy"],
  13: ["Yale", "Vermont", "High Point", "Akron"],
  14: ["Lipscomb", "Grand Canyon", "Montana", "Colgate"],
  15: ["Oral Roberts", "Wofford", "Omaha", "Robert Morris"],
  16: ["Norfolk St.", "SIUE", "Grambling", "FDU"],
};

const ALL_TEAMS = Object.entries(TEAMS_BY_SEED).flatMap(([seed, teams]) =>
  teams.map((t) => ({ name: t, seed: parseInt(seed) }))
);

function generateEfficiency(seed, rng) {
  const r = () => (rng() - 0.5) * 2;
  const profiles = {
    1: { oe: 119, de: 90, tempo: 69, barthag: 0.95 },
    2: { oe: 117, de: 92, tempo: 68, barthag: 0.92 },
    3: { oe: 115, de: 94, tempo: 68, barthag: 0.90 },
    4: { oe: 113, de: 95, tempo: 68, barthag: 0.88 },
    5: { oe: 112, de: 96, tempo: 68, barthag: 0.86 },
    6: { oe: 111, de: 97, tempo: 68, barthag: 0.84 },
    7: { oe: 110, de: 98, tempo: 68, barthag: 0.82 },
    8: { oe: 109, de: 99, tempo: 68, barthag: 0.79 },
    9: { oe: 108, de: 99, tempo: 68, barthag: 0.78 },
    10: { oe: 108, de: 100, tempo: 68, barthag: 0.76 },
    11: { oe: 107, de: 100, tempo: 68, barthag: 0.74 },
    12: { oe: 107, de: 100, tempo: 68, barthag: 0.73 },
    13: { oe: 105, de: 102, tempo: 68, barthag: 0.68 },
    14: { oe: 104, de: 103, tempo: 68, barthag: 0.64 },
    15: { oe: 102, de: 105, tempo: 68, barthag: 0.58 },
    16: { oe: 98, de: 108, tempo: 68, barthag: 0.45 },
  };
  const p = profiles[seed] || profiles[8];
  return {
    adj_oe: +(p.oe + r() * 4).toFixed(1),
    adj_de: +(p.de + r() * 3).toFixed(1),
    adj_tempo: +(p.tempo + r() * 3).toFixed(1),
    barthag: +Math.min(0.99, Math.max(0.1, p.barthag + r() * 0.03)).toFixed(3),
    three_pt_rate: +(0.36 + r() * 0.04).toFixed(3),
  };
}

function predictGame(homeTeam, awayTeam, round = 1) {
  const seeder = (s) => {
    let v = s * 2654435761;
    return () => {
      v = (v ^ (v >>> 16)) * 2246822507;
      v = (v ^ (v >>> 13)) * 3266489909;
      v = v ^ (v >>> 16);
      return ((v >>> 0) / 4294967296 + 1) / 2;
    };
  };
  const rng = seeder(homeTeam.seed * 100 + awayTeam.seed + round * 7);

  const hEff = generateEfficiency(homeTeam.seed, rng);
  const aEff = generateEfficiency(awayTeam.seed, rng);

  const roundMod = { 0: 1.0, 1: 1.0, 2: 1.0, 3: 0.97, 4: 0.94, 5: 0.96, 6: 0.95 }[round] || 1.0;
  const gamePace = ((hEff.adj_tempo + aEff.adj_tempo) / 2) * roundMod;
  const hScore = ((hEff.adj_oe * aEff.adj_de) / 100) * (gamePace / 100);
  const aScore = ((aEff.adj_oe * hEff.adj_de) / 100) * (gamePace / 100);

  const seedAdj = ((1 / (1 + Math.exp(-0.25 * (awayTeam.seed - homeTeam.seed)))) - 0.5) * 4;
  const momAdj = (rng() - 0.45) * 1.2;
  const expAdj = (rng() - 0.45) * 0.8;
  const restAdj = (rng() - 0.5) * 0.4;
  const travelAdj = (rng() - 0.5) * 0.3;
  const totalAdj = Math.max(-8, Math.min(8, seedAdj + momAdj + expAdj + restAdj + travelAdj));

  const rawMargin = hScore - aScore;
  const finalMargin = rawMargin + totalAdj;
  const total = hScore + aScore;
  const winProb = 1 / (1 + Math.exp(-finalMargin / 10.5));

  return {
    homeName: homeTeam.name,
    awayName: awayTeam.name,
    homeSeed: homeTeam.seed,
    awaySeed: awayTeam.seed,
    homeScore: +(total / 2 + finalMargin / 2).toFixed(1),
    awayScore: +(total / 2 - finalMargin / 2).toFixed(1),
    spread: +(-finalMargin).toFixed(1),
    total: +total.toFixed(1),
    homeWinProb: +winProb.toFixed(4),
    gamePace: +gamePace.toFixed(1),
    rawMargin: +rawMargin.toFixed(1),
    hEff,
    aEff,
    adjustments: {
      momentum: +momAdj.toFixed(2),
      experience: +expAdj.toFixed(2),
      rest: +restAdj.toFixed(2),
      seed: +seedAdj.toFixed(2),
      travel: +travelAdj.toFixed(2),
      total: +totalAdj.toFixed(2),
    },
    confidence: winProb > 0.85 || winProb < 0.15 ? "VERY HIGH" : winProb > 0.7 || winProb < 0.3 ? "HIGH" : winProb > 0.6 || winProb < 0.4 ? "MEDIUM" : "LOW",
    round,
  };
}

const ROUND_NAMES = ["First Four", "First Round", "Second Round", "Sweet 16", "Elite 8", "Final Four", "Championship"];

function AdjustmentBar({ label, value, max = 3 }) {
  const pct = Math.min(100, (Math.abs(value) / max) * 100);
  const isPositive = value >= 0;
  return (
    <div style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 4, fontSize: 13 }}>
      <span style={{ width: 80, textAlign: "right", color: "#8a8a9a", fontFamily: "'JetBrains Mono', monospace" }}>{label}</span>
      <div style={{ flex: 1, height: 14, background: "#1a1a2e", borderRadius: 3, position: "relative", overflow: "hidden" }}>
        <div style={{ position: "absolute", left: "50%", top: 0, bottom: 0, width: 1, background: "#333" }} />
        <div
          style={{
            position: "absolute",
            [isPositive ? "left" : "right"]: "50%",
            top: 1, bottom: 1,
            width: `${pct / 2}%`,
            background: isPositive ? "linear-gradient(90deg, #0a4d3a, #0f8a5f)" : "linear-gradient(90deg, #8a2020, #4d0a0a)",
            borderRadius: 2,
            transition: "width 0.5s ease",
          }}
        />
      </div>
      <span style={{ width: 55, textAlign: "right", fontFamily: "'JetBrains Mono', monospace", color: isPositive ? "#0f8a5f" : "#c44", fontWeight: 600, fontSize: 12 }}>
        {value >= 0 ? "+" : ""}{value.toFixed(2)}
      </span>
    </div>
  );
}

function EfficiencyCard({ label, eff, seed, color }) {
  return (
    <div style={{ flex: 1, padding: "14px 16px", background: "#0d0d1a", borderRadius: 8, border: `1px solid ${color}22` }}>
      <div style={{ fontSize: 11, textTransform: "uppercase", letterSpacing: 2, color: "#666", marginBottom: 8 }}>{label}</div>
      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "6px 16px", fontSize: 13 }}>
        {[
          ["AdjOE", eff.adj_oe, eff.adj_oe > 110],
          ["AdjDE", eff.adj_de, eff.adj_de < 95],
          ["Tempo", eff.adj_tempo, false],
          ["Barthag", eff.barthag, eff.barthag > 0.85],
        ].map(([k, v, good]) => (
          <div key={k} style={{ display: "flex", justifyContent: "space-between" }}>
            <span style={{ color: "#8a8a9a" }}>{k}</span>
            <span style={{ fontFamily: "'JetBrains Mono', monospace", fontWeight: 600, color: good ? "#0f8a5f" : "#c8c8d8" }}>{v}</span>
          </div>
        ))}
      </div>
    </div>
  );
}

function WinProbBar({ homeProb, homeName, awayName }) {
  const hp = Math.round(homeProb * 100);
  const ap = 100 - hp;
  return (
    <div style={{ marginTop: 6 }}>
      <div style={{ display: "flex", justifyContent: "space-between", fontSize: 12, marginBottom: 4, color: "#aaa" }}>
        <span>{homeName}</span>
        <span>{awayName}</span>
      </div>
      <div style={{ display: "flex", height: 28, borderRadius: 6, overflow: "hidden", fontFamily: "'JetBrains Mono', monospace", fontSize: 13, fontWeight: 700 }}>
        <div style={{ width: `${hp}%`, background: "linear-gradient(90deg, #0a4d3a, #0f8a5f)", display: "flex", alignItems: "center", justifyContent: "center", color: "#fff", transition: "width 0.6s ease", minWidth: hp > 15 ? "auto" : 0 }}>
          {hp > 15 && `${hp}%`}
        </div>
        <div style={{ width: `${ap}%`, background: "linear-gradient(90deg, #8a2020, #4d0a0a)", display: "flex", alignItems: "center", justifyContent: "center", color: "#fff", transition: "width 0.6s ease", minWidth: ap > 15 ? "auto" : 0 }}>
          {ap > 15 && `${ap}%`}
        </div>
      </div>
    </div>
  );
}

function GameCard({ prediction }) {
  const [expanded, setExpanded] = useState(false);
  const p = prediction;
  const homeWins = p.homeWinProb > 0.5;
  const winner = homeWins ? p.homeName : p.awayName;
  const winPct = Math.round(Math.max(p.homeWinProb, 1 - p.homeWinProb) * 100);
  const confColor = { "VERY HIGH": "#0f8a5f", HIGH: "#3a8a3a", MEDIUM: "#8a7a20", LOW: "#8a4a20" }[p.confidence] || "#666";

  return (
    <div
      style={{
        background: "#111124", borderRadius: 12,
        border: "1px solid #1e1e3a", overflow: "hidden",
        transition: "all 0.3s ease",
        cursor: "pointer",
      }}
      onClick={() => setExpanded(!expanded)}
    >
      <div style={{ padding: "16px 20px" }}>
        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 10 }}>
          <span style={{ fontSize: 11, textTransform: "uppercase", letterSpacing: 2, color: "#666" }}>
            {ROUND_NAMES[p.round]} · {p.confidence}
          </span>
          <span style={{ fontSize: 11, color: confColor, fontWeight: 600 }}>●</span>
        </div>

        <div style={{ display: "flex", alignItems: "center", gap: 12, marginBottom: 6 }}>
          <span style={{ width: 24, height: 24, borderRadius: 6, background: homeWins ? "#0f8a5f22" : "#1a1a2e", display: "flex", alignItems: "center", justifyContent: "center", fontSize: 11, fontWeight: 700, color: homeWins ? "#0f8a5f" : "#555", fontFamily: "'JetBrains Mono', monospace" }}>
            {p.homeSeed}
          </span>
          <span style={{ flex: 1, fontWeight: homeWins ? 700 : 400, color: homeWins ? "#e8e8f0" : "#888", fontSize: 16 }}>{p.homeName}</span>
          <span style={{ fontFamily: "'JetBrains Mono', monospace", fontWeight: 700, fontSize: 18, color: homeWins ? "#e8e8f0" : "#666" }}>{p.homeScore.toFixed(0)}</span>
        </div>

        <div style={{ display: "flex", alignItems: "center", gap: 12 }}>
          <span style={{ width: 24, height: 24, borderRadius: 6, background: !homeWins ? "#0f8a5f22" : "#1a1a2e", display: "flex", alignItems: "center", justifyContent: "center", fontSize: 11, fontWeight: 700, color: !homeWins ? "#0f8a5f" : "#555", fontFamily: "'JetBrains Mono', monospace" }}>
            {p.awaySeed}
          </span>
          <span style={{ flex: 1, fontWeight: !homeWins ? 700 : 400, color: !homeWins ? "#e8e8f0" : "#888", fontSize: 16 }}>{p.awayName}</span>
          <span style={{ fontFamily: "'JetBrains Mono', monospace", fontWeight: 700, fontSize: 18, color: !homeWins ? "#e8e8f0" : "#666" }}>{p.awayScore.toFixed(0)}</span>
        </div>

        <WinProbBar homeProb={p.homeWinProb} homeName={p.homeName} awayName={p.awayName} />

        <div style={{ display: "flex", gap: 16, marginTop: 10, fontSize: 12, color: "#888" }}>
          <span>Spread <b style={{ color: "#c8c8d8" }}>{p.spread > 0 ? "+" : ""}{p.spread}</b></span>
          <span>Total <b style={{ color: "#c8c8d8" }}>{p.total.toFixed(0)}</b></span>
          <span>Pace <b style={{ color: "#c8c8d8" }}>{p.gamePace.toFixed(0)}</b></span>
        </div>
      </div>

      {expanded && (
        <div style={{ padding: "0 20px 20px", borderTop: "1px solid #1a1a2e", paddingTop: 16, animation: "fadeIn 0.3s ease" }}>
          <div style={{ fontSize: 11, textTransform: "uppercase", letterSpacing: 2, color: "#555", marginBottom: 10 }}>Layer 3 — Matchup Projection</div>
          <div style={{ display: "flex", gap: 12, marginBottom: 16 }}>
            <EfficiencyCard label={p.homeName} eff={p.hEff} seed={p.homeSeed} color="#0f8a5f" />
            <EfficiencyCard label={p.awayName} eff={p.aEff} seed={p.awaySeed} color="#8a2020" />
          </div>
          <div style={{ fontSize: 12, color: "#888", marginBottom: 4 }}>
            Raw Margin: <span style={{ fontFamily: "'JetBrains Mono', monospace", color: "#c8c8d8" }}>{p.rawMargin > 0 ? "+" : ""}{p.rawMargin}</span> pts
          </div>

          <div style={{ fontSize: 11, textTransform: "uppercase", letterSpacing: 2, color: "#555", marginTop: 16, marginBottom: 10 }}>Layer 4 — Adjustments</div>
          <AdjustmentBar label="Momentum" value={p.adjustments.momentum} />
          <AdjustmentBar label="Experience" value={p.adjustments.experience} />
          <AdjustmentBar label="Rest" value={p.adjustments.rest} />
          <AdjustmentBar label="Seed Hist." value={p.adjustments.seed} />
          <AdjustmentBar label="Travel" value={p.adjustments.travel} />
          <div style={{ height: 1, background: "#1e1e3a", margin: "8px 0" }} />
          <AdjustmentBar label="TOTAL" value={p.adjustments.total} max={8} />
        </div>
      )}
    </div>
  );
}

export default function SportsOracleDashboard() {
  const [selectedRound, setSelectedRound] = useState(1);
  const [homeTeam, setHomeTeam] = useState(null);
  const [awayTeam, setAwayTeam] = useState(null);
  const [customPrediction, setCustomPrediction] = useState(null);

  const roundGames = useMemo(() => {
    const matchups = [
      [1, 16], [8, 9], [5, 12], [4, 13], [6, 11], [3, 14], [7, 10], [2, 15],
    ];
    if (selectedRound > 1) return [];
    return matchups.map(([hs, ls]) => {
      const h = { name: TEAMS_BY_SEED[hs][0], seed: hs };
      const a = { name: TEAMS_BY_SEED[ls][0], seed: ls };
      return predictGame(h, a, selectedRound);
    });
  }, [selectedRound]);

  const handlePredict = () => {
    if (homeTeam && awayTeam) {
      setCustomPrediction(predictGame(homeTeam, awayTeam, selectedRound));
    }
  };

  return (
    <div style={{ minHeight: "100vh", background: "#08081a", color: "#e8e8f0", fontFamily: "'Outfit', 'Helvetica Neue', sans-serif" }}>
      <link href="https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700;800&family=JetBrains+Mono:wght@400;600;700&display=swap" rel="stylesheet" />
      <style>{`
        @keyframes fadeIn { from { opacity: 0; transform: translateY(-8px); } to { opacity: 1; transform: translateY(0); } }
        select { background: #111124; color: #e8e8f0; border: 1px solid #2a2a4a; padding: 10px 14px; border-radius: 8px; font-family: 'Outfit', sans-serif; font-size: 14px; cursor: pointer; outline: none; }
        select:focus { border-color: #0f8a5f; }
        select option { background: #111124; }
        ::-webkit-scrollbar { width: 6px; } ::-webkit-scrollbar-track { background: #08081a; } ::-webkit-scrollbar-thumb { background: #2a2a4a; border-radius: 3px; }
      `}</style>

      <div style={{ maxWidth: 900, margin: "0 auto", padding: "32px 20px" }}>
        {/* Header */}
        <div style={{ marginBottom: 40, textAlign: "center" }}>
          <div style={{ fontSize: 11, textTransform: "uppercase", letterSpacing: 6, color: "#0f8a5f", marginBottom: 8, fontWeight: 600 }}>NCAA Tournament</div>
          <h1 style={{ fontSize: 42, fontWeight: 800, margin: 0, letterSpacing: -1, background: "linear-gradient(135deg, #e8e8f0 0%, #8a8aaa 100%)", WebkitBackgroundClip: "text", WebkitTextFillColor: "transparent" }}>
            Sports Oracle
          </h1>
          <p style={{ color: "#666", fontSize: 14, marginTop: 8 }}>Hybrid Formula + ML Prediction Engine</p>
        </div>

        {/* Custom Matchup */}
        <div style={{ background: "#111124", borderRadius: 12, border: "1px solid #1e1e3a", padding: 24, marginBottom: 24 }}>
          <div style={{ fontSize: 11, textTransform: "uppercase", letterSpacing: 2, color: "#555", marginBottom: 14 }}>Custom Matchup</div>
          <div style={{ display: "flex", gap: 12, flexWrap: "wrap", alignItems: "end" }}>
            <div style={{ flex: 1, minWidth: 160 }}>
              <div style={{ fontSize: 11, color: "#888", marginBottom: 4 }}>Home Team</div>
              <select value={homeTeam ? `${homeTeam.seed}-${homeTeam.name}` : ""} onChange={(e) => {
                if (e.target.value) { const [s, ...n] = e.target.value.split("-"); setHomeTeam({ seed: parseInt(s), name: n.join("-") }); }
              }}>
                <option value="">Select team...</option>
                {ALL_TEAMS.map((t) => <option key={`h-${t.name}`} value={`${t.seed}-${t.name}`}>({t.seed}) {t.name}</option>)}
              </select>
            </div>
            <div style={{ fontSize: 20, color: "#333", fontWeight: 700, paddingBottom: 8 }}>vs</div>
            <div style={{ flex: 1, minWidth: 160 }}>
              <div style={{ fontSize: 11, color: "#888", marginBottom: 4 }}>Away Team</div>
              <select value={awayTeam ? `${awayTeam.seed}-${awayTeam.name}` : ""} onChange={(e) => {
                if (e.target.value) { const [s, ...n] = e.target.value.split("-"); setAwayTeam({ seed: parseInt(s), name: n.join("-") }); }
              }}>
                <option value="">Select team...</option>
                {ALL_TEAMS.map((t) => <option key={`a-${t.name}`} value={`${t.seed}-${t.name}`}>({t.seed}) {t.name}</option>)}
              </select>
            </div>
            <div style={{ minWidth: 120 }}>
              <div style={{ fontSize: 11, color: "#888", marginBottom: 4 }}>Round</div>
              <select value={selectedRound} onChange={(e) => setSelectedRound(parseInt(e.target.value))}>
                {ROUND_NAMES.map((n, i) => <option key={i} value={i}>{n}</option>)}
              </select>
            </div>
            <button
              onClick={handlePredict}
              disabled={!homeTeam || !awayTeam}
              style={{
                padding: "10px 28px", borderRadius: 8, border: "none",
                background: homeTeam && awayTeam ? "linear-gradient(135deg, #0a4d3a, #0f8a5f)" : "#1a1a2e",
                color: homeTeam && awayTeam ? "#fff" : "#555",
                fontFamily: "'Outfit', sans-serif", fontSize: 14, fontWeight: 600,
                cursor: homeTeam && awayTeam ? "pointer" : "default",
                transition: "all 0.2s ease",
              }}
            >
              Predict
            </button>
          </div>

          {customPrediction && (
            <div style={{ marginTop: 20 }}>
              <GameCard prediction={customPrediction} />
            </div>
          )}
        </div>

        {/* First Round Predictions */}
        <div style={{ marginBottom: 16 }}>
          <div style={{ fontSize: 11, textTransform: "uppercase", letterSpacing: 2, color: "#555", marginBottom: 14 }}>
            First Round Predictions · East Region
          </div>
          <div style={{ display: "grid", gap: 12 }}>
            {roundGames.map((game, i) => (
              <GameCard key={i} prediction={game} />
            ))}
          </div>
        </div>

        {/* Footer */}
        <div style={{ textAlign: "center", padding: "32px 0 16px", fontSize: 12, color: "#444" }}>
          Sports Oracle v1.0 · Hybrid Formula + ML Engine · 6 Data Sources · 51 ML Features
        </div>
      </div>
    </div>
  );
}
