#!/usr/bin/env node
/**
 * sync-sheets.js — Push CSV trade files to Google Sheets
 * 
 * Reads all CSV files from logs/paper/<strategy>/<strategy>.csv
 * and logs/live/<strategy>/<strategy>.csv, then updates the
 * corresponding sheet tab in the Google Sheet.
 * 
 * Usage: node scripts/sync-sheets.js
 * No AI, no tokens — pure API calls.
 */

const {google} = require('googleapis');
const fs = require('fs');
const path = require('path');

const SPREADSHEET_ID = '1xJWy6ZLCATaB_RMbe0qKy7ur_dc4_RxZR0TiqR-as80';
const SERVICE_ACCOUNT_PATH = path.join(__dirname, '..', 'service_account.json');
const LOGS_DIR = path.join(__dirname, '..', 'logs');

async function main() {
  if (!fs.existsSync(SERVICE_ACCOUNT_PATH)) {
    console.error('❌ service_account.json not found');
    process.exit(1);
  }

  const creds = JSON.parse(fs.readFileSync(SERVICE_ACCOUNT_PATH));
  const auth = new google.auth.GoogleAuth({
    credentials: creds,
    scopes: ['https://www.googleapis.com/auth/spreadsheets'],
  });
  const client = await auth.getClient();
  const sheets = google.sheets({version: 'v4', auth: client});

  // Get existing sheet tabs
  const existing = await sheets.spreadsheets.get({ spreadsheetId: SPREADSHEET_ID });
  const existingTabs = existing.data.sheets.map(s => s.properties.title);

  // Find all CSV files in logs/paper/ and logs/live/
  const csvFiles = [];
  for (const mode of ['paper', 'live']) {
    const modeDir = path.join(LOGS_DIR, mode);
    if (!fs.existsSync(modeDir)) continue;
    
    for (const strategy of fs.readdirSync(modeDir)) {
      const stratDir = path.join(modeDir, strategy);
      if (!fs.statSync(stratDir).isDirectory()) continue;
      
      const csvPath = path.join(stratDir, `${strategy}.csv`);
      if (fs.existsSync(csvPath)) {
        // Tab name: strategy name for paper, "live_<strategy>" for live
        const tabName = mode === 'paper' ? strategy : `live_${strategy}`;
        // Skip combined strategies — not useful in sheets
        if (strategy.startsWith('combined_')) continue;
        csvFiles.push({ path: csvPath, tabName, strategy, mode });
      }
    }
  }

  // Also include all_rounds.csv if it exists
  const allRoundsCsv = path.join(LOGS_DIR, 'rounds', 'all_rounds.csv');
  if (fs.existsSync(allRoundsCsv)) {
    csvFiles.push({ path: allRoundsCsv, tabName: 'all_rounds', strategy: 'all_rounds', mode: 'rounds' });
  }

  // Also include pool_snapshots.csv if it exists
  const poolSnapshotsCsv = path.join(LOGS_DIR, 'rounds', 'pool_snapshots.csv');
  if (fs.existsSync(poolSnapshotsCsv)) {
    csvFiles.push({ path: poolSnapshotsCsv, tabName: 'pool_snapshots', strategy: 'pool_snapshots', mode: 'rounds' });
  }

  if (csvFiles.length === 0) {
    console.log('No CSV files found to sync.');
    return;
  }

  console.log(`Found ${csvFiles.length} CSV file(s) to sync.`);

  for (const {path: csvPath, tabName} of csvFiles) {
    // Read and parse CSV
    const raw = fs.readFileSync(csvPath, 'utf-8');
    const lines = raw.trim().split('\n');
    const rows = lines.map(line => {
      // Simple CSV parse (handles commas in values if quoted)
      const result = [];
      let current = '';
      let inQuotes = false;
      for (const char of line) {
        if (char === '"') { inQuotes = !inQuotes; continue; }
        if (char === ',' && !inQuotes) { result.push(current); current = ''; continue; }
        current += char;
      }
      result.push(current);
      return result;
    });

    if (rows.length < 2) {
      console.log(`  ⏸ ${tabName}: no data rows, skipping`);
      continue;
    }

    // Create tab if it doesn't exist (case-insensitive check)
    const tabExists = existingTabs.some(t => t.toLowerCase() === tabName.toLowerCase());
    if (!tabExists) {
      await sheets.spreadsheets.batchUpdate({
        spreadsheetId: SPREADSHEET_ID,
        requestBody: {
          requests: [{ addSheet: { properties: { title: tabName } } }],
        },
      });
      existingTabs.push(tabName);
      console.log(`  📄 Created tab: ${tabName}`);
    }

    // Find actual tab name (may differ in case)
    const actualTab = existingTabs.find(t => t.toLowerCase() === tabName.toLowerCase()) || tabName;

    // Clear existing data and write new
    await sheets.spreadsheets.values.clear({
      spreadsheetId: SPREADSHEET_ID,
      range: `'${actualTab}'!A:Z`,
    });

    await sheets.spreadsheets.values.update({
      spreadsheetId: SPREADSHEET_ID,
      range: `'${actualTab}'!A1`,
      valueInputOption: 'USER_ENTERED',
      requestBody: { values: rows },
    });

    console.log(`  ✅ ${actualTab}: ${rows.length - 1} trades synced`);
  }

  // Refresh all analysis tabs (add delays to avoid write quota)
  await sleep(2000);
  await refreshEpochMap(sheets, existingTabs);
  await sleep(2000);
  await refreshStrategyComparison(sheets, existingTabs);
  await sleep(2000);
  await refreshDeepEdge(sheets, existingTabs);
  await sleep(2000);
  await refreshCrowdAccuracy(sheets, existingTabs);

  console.log('Done.');
}


// ═══════════════════════════════════════════════════════════════════════════
// 🏆 Strategy Comparison — fully dynamic, computed from data
// ═══════════════════════════════════════════════════════════════════════════

async function refreshStrategyComparison(sheets, existingTabs) {
  const SSID = SPREADSHEET_ID;
  const analysisTabs = ['🏆 Strategy Comparison', '🔗 Cross-Strategy (Epoch)', '📊 Deep Edge Analysis', '🎯 Optimal Filters', '📈 PnL Curves', '🕐 Time Heatmap', '👥 Crowd Accuracy'];
  const STRATS = existingTabs.filter(t => !analysisTabs.includes(t) && !t.startsWith('combined_') && t !== 'all_rounds' && t !== 'pool_snapshots');
  const tab = existingTabs.find(t => t.includes('Strategy Comparison'));
  if (!tab) { console.log('  ⏸ Strategy Comparison tab not found'); return; }

  // Global drift/timing accumulators
  let globalDriftSum = 0, globalDriftCount = 0;
  let globalDrift5Count = 0, globalDrift10Count = 0;
  let globalTimingSum = 0, globalTimingCount = 0;

  // Load all trade data
  const results = [];
  for (const s of STRATS) {
    const actualTab = existingTabs.find(t => t.toLowerCase() === s.toLowerCase()) || s;
    try {
      const res = await sheets.spreadsheets.values.get({ spreadsheetId: SSID, range: `'${actualTab}'!A1:AA5000` });
      const rows = res.data.values || [];
      if (rows.length < 2) continue;
      const h = rows[0];
      const iSide = h.indexOf('side_label'), iOutcome = h.indexOf('outcome');
      const iEdge = h.indexOf('edge_at_entry'), iPnl = h.indexOf('pnl_usdc');
      const iPos = h.indexOf('position_size_usdc'), iPayout = h.indexOf('payout_per_share');
      const iDrift = h.indexOf('pool_drift_pct');
      const iEntryTs = h.indexOf('timestamp_entry');
      const iWindowEnd = h.indexOf('window_end_ts');

      const trades = [];
      for (let i = 1; i < rows.length; i++) {
        const r = rows[i];
        const outcome = r[iOutcome] || '';
        if (outcome !== 'WIN' && outcome !== 'LOSS') continue;
        trades.push({
          side: r[iSide] || '',
          outcome,
          edge: parseFloat(r[iEdge]) || 0,
          pnl: parseFloat(r[iPnl]) || 0,
          pos: parseFloat(r[iPos]) || 0,
          payout: parseFloat(r[iPayout]) || 0,
          drift: iDrift >= 0 ? (parseFloat(r[iDrift]) || 0) : -1,
          entryTs: iEntryTs >= 0 ? (parseFloat(r[iEntryTs]) || 0) : 0,
          windowEnd: iWindowEnd >= 0 ? (parseFloat(r[iWindowEnd]) || 0) : 0,
        });
      }
      if (trades.length === 0) continue;

      const wins = trades.filter(t => t.outcome === 'WIN');
      const losses = trades.filter(t => t.outcome === 'LOSS');
      const total = trades.length;
      const wr = total > 0 ? wins.length / total * 100 : 0;
      const pnl = trades.reduce((s, t) => s + t.pnl, 0);
      const avgEdge = trades.reduce((s, t) => s + t.edge, 0) / total;
      const upTrades = trades.filter(t => t.side === 'UP');
      const downTrades = trades.filter(t => t.side === 'DOWN');
      const upWr = upTrades.length > 0 ? upTrades.filter(t => t.outcome === 'WIN').length / upTrades.length * 100 : 0;
      const downWr = downTrades.length > 0 ? downTrades.filter(t => t.outcome === 'WIN').length / downTrades.length * 100 : 0;
      const avgWinPnl = wins.length > 0 ? wins.reduce((s, t) => s + t.pnl, 0) / wins.length : 0;
      const avgLossPnl = losses.length > 0 ? losses.reduce((s, t) => s + t.pnl, 0) / losses.length : 0;
      const rr = avgLossPnl !== 0 ? Math.abs(avgWinPnl / avgLossPnl) : 0;
      const wagered = trades.reduce((s, t) => s + t.pos, 0);
      const roi = wagered > 0 ? pnl / wagered * 100 : 0;

      // Drift stats — only trades where drift > 0 (data available)
      const driftTrades = iDrift >= 0 ? trades.filter(t => t.drift > 0) : [];
      const hasDrift = driftTrades.length > 0;
      const avgDrift = hasDrift ? r2(driftTrades.reduce((s, t) => s + t.drift, 0) / driftTrades.length * 100) : null;
      const pctDrift5 = hasDrift ? r2(driftTrades.filter(t => t.drift > 0.05).length / driftTrades.length * 100) : null;
      const pctDrift10 = hasDrift ? r2(driftTrades.filter(t => t.drift > 0.10).length / driftTrades.length * 100) : null;

      // Accumulate global drift stats
      if (hasDrift) {
        globalDriftSum += driftTrades.reduce((s, t) => s + t.drift, 0);
        globalDriftCount += driftTrades.length;
        globalDrift5Count += driftTrades.filter(t => t.drift > 0.05).length;
        globalDrift10Count += driftTrades.filter(t => t.drift > 0.10).length;
      }

      // Timing: seconds before lock (window_end_ts - timestamp_entry)
      if (iEntryTs >= 0 && iWindowEnd >= 0) {
        for (const t of trades) {
          if (t.entryTs > 0 && t.windowEnd > 0) {
            let diff = t.windowEnd - t.entryTs;
            if (t.windowEnd > 1e10) diff = diff / 1000; // ms → s
            if (diff > 0 && diff < 3600) {
              globalTimingSum += diff;
              globalTimingCount++;
            }
          }
        }
      }

      results.push([
        s, total, wins.length, losses.length,
        r2(wr), r2(pnl), r2(pnl / total), r4(avgEdge),
        upTrades.length, downTrades.length,
        r2(upWr), r2(downWr),
        r2(avgWinPnl), r2(avgLossPnl), r2(rr), r2(roi),
        avgDrift !== null ? avgDrift : 'N/A',
        pctDrift5 !== null ? pctDrift5 : 'N/A',
        pctDrift10 !== null ? pctDrift10 : 'N/A',
      ]);
    } catch(e) {
      console.log(`  ⚠️ Comparison: could not load ${s}: ${e.message}`);
    }
  }

  // Build trading quality block (global)
  const globalAvgDrift = globalDriftCount > 0 ? r2(globalDriftSum / globalDriftCount * 100) + '%' : 'N/A';
  const globalPctDrift5 = globalDriftCount > 0 ? r2(globalDrift5Count / globalDriftCount * 100) + '%' : 'N/A';
  const globalPctDrift10 = globalDriftCount > 0 ? r2(globalDrift10Count / globalDriftCount * 100) + '%' : 'N/A';
  const globalAvgTiming = globalTimingCount > 0 ? r2(globalTimingSum / globalTimingCount) + 's' : 'N/A';

  const syncTimestamp = new Date().toLocaleString('en-CH', { timeZone: 'Europe/Zurich', dateStyle: 'short', timeStyle: 'medium' });

  const n = results.length;
  const outputRows = [
    ['📊 TRADING QUALITY (global)'],
    ['Last sync:', syncTimestamp],
    ['Avg Pool Drift:', globalAvgDrift],
    ['Trades with drift >5%:', globalPctDrift5],
    ['Trades with drift >10%:', globalPctDrift10],
    ['Avg seconds before lock:', globalAvgTiming],
    [],
    [`🏆 Strategy Comparison — ${n} Strategies`],
    ['WIN+LOSS only. Updated hourly by sync script.'],
    [],
    ['Strategy', 'Trades (W+L)', 'Wins', 'Losses', 'Win Rate %', 'Total PnL ($)',
     'Avg PnL/Trade ($)', 'Avg Edge', 'UP Trades', 'DOWN Trades', 'UP WR %', 'DOWN WR %',
     'Avg WIN PnL ($)', 'Avg LOSS PnL ($)', 'R/R Ratio', 'ROI %',
     'Avg Drift %', 'Drift >5%', 'Drift >10%'],
    ...results.sort((a, b) => b[4] - a[4]),  // Sort by Win Rate % descending
    [],
    ['📊 COMBINED'],
  ];

  // Combined row
  const totTrades = results.reduce((s, r) => s + r[1], 0);
  const totWins = results.reduce((s, r) => s + r[2], 0);
  const totLosses = results.reduce((s, r) => s + r[3], 0);
  const totPnl = results.reduce((s, r) => s + r[5], 0);
  const totWr = totTrades > 0 ? totWins / totTrades * 100 : 0;
  outputRows.push(['ALL', totTrades, totWins, totLosses, r2(totWr), r2(totPnl), r2(totTrades > 0 ? totPnl / totTrades : 0)]);

  // Best/worst (min 30 trades)
  const eligible = results.filter(r => r[1] >= 30);
  if (eligible.length > 0) {
    const bestWr = eligible.reduce((b, r) => r[4] > b[4] ? r : b);
    const bestPnl = eligible.reduce((b, r) => r[5] > b[5] ? r : b);
    const worstPnl = eligible.reduce((b, r) => r[5] < b[5] ? r : b);
    outputRows.push(['🥇 Best Win Rate (min 30)', bestWr[0], bestWr[4] + '%']);
    outputRows.push(['🥇 Best PnL (min 30)', bestPnl[0], '$' + bestPnl[5]]);
    outputRows.push(['❌ Worst PnL (min 30)', worstPnl[0], '$' + worstPnl[5]]);
  } else {
    outputRows.push(['🥇 Best Win Rate (min 30)', 'Not enough data']);
    outputRows.push(['🥇 Best PnL (min 30)', 'Not enough data']);
    outputRows.push(['❌ Worst PnL (min 30)', 'Not enough data']);
  }
  const worstAll = results.length > 0 ? results.reduce((b, r) => r[5] < b[5] ? r : b) : null;
  outputRows.push(['❌ Worst PnL ($)', worstAll ? worstAll[0] : 'N/A', worstAll ? '$' + worstAll[5] : 'N/A']);

  await sheets.spreadsheets.values.clear({ spreadsheetId: SSID, range: `'${tab}'!A1:S70` });
  await sheets.spreadsheets.values.update({
    spreadsheetId: SSID, range: `'${tab}'!A1`,
    valueInputOption: 'USER_ENTERED',
    requestBody: { values: outputRows },
  });

  console.log(`  ✅ Strategy Comparison refreshed: ${n} strategies`);
}


// ═══════════════════════════════════════════════════════════════════════════
// 📊 Deep Edge Analysis — fully dynamic, computed from data
// ═══════════════════════════════════════════════════════════════════════════

async function refreshDeepEdge(sheets, existingTabs) {
  const SSID = SPREADSHEET_ID;
  const analysisTabs = ['🏆 Strategy Comparison', '🔗 Cross-Strategy (Epoch)', '📊 Deep Edge Analysis', '🎯 Optimal Filters', '📈 PnL Curves', '🕐 Time Heatmap', '👥 Crowd Accuracy'];
  const STRATS = existingTabs.filter(t => !analysisTabs.includes(t) && !t.startsWith('combined_') && t !== 'all_rounds' && t !== 'pool_snapshots');
  const tab = existingTabs.find(t => t.includes('Deep Edge'));
  if (!tab) { console.log('  ⏸ Deep Edge tab not found'); return; }

  const buckets = [[0, 0.1], [0.1, 0.2], [0.2, 0.3], [0.3, 0.4], [0.4, 0.5], [0.5, 1.0]];
  const outputRows = [];
  let stratCount = 0;

  for (const s of STRATS) {
    const actualTab = existingTabs.find(t => t.toLowerCase() === s.toLowerCase()) || s;
    try {
      const res = await sheets.spreadsheets.values.get({ spreadsheetId: SSID, range: `'${actualTab}'!A1:AA5000` });
      const rows = res.data.values || [];
      if (rows.length < 2) continue;
      const h = rows[0];
      const iOutcome = h.indexOf('outcome'), iEdge = h.indexOf('edge_at_entry');
      const iPnl = h.indexOf('pnl_usdc'), iPayout = h.indexOf('payout_per_share');

      const trades = [];
      for (let i = 1; i < rows.length; i++) {
        const r = rows[i];
        const outcome = r[iOutcome] || '';
        if (outcome !== 'WIN' && outcome !== 'LOSS') continue;
        trades.push({
          outcome,
          edge: parseFloat(r[iEdge]) || 0,
          pnl: parseFloat(r[iPnl]) || 0,
          payout: parseFloat(r[iPayout]) || 0,
        });
      }
      if (trades.length === 0) continue;

      stratCount++;
      outputRows.push([`═══ ${s} ═══`]);
      outputRows.push(['Edge Bucket', 'Trades', 'Wins', 'Win Rate %', 'PnL ($)', 'Avg Payout (win)']);

      for (const [lo, hi] of buckets) {
        const bucket = trades.filter(t => t.edge >= lo && t.edge < hi);
        const bWins = bucket.filter(t => t.outcome === 'WIN');
        const bTotal = bucket.length;
        const bWr = bTotal > 0 ? bWins.length / bTotal * 100 : 0;
        const bPnl = bucket.reduce((s, t) => s + t.pnl, 0);
        const bAvgPayout = bWins.length > 0 ? bWins.reduce((s, t) => s + t.payout, 0) / bWins.length : 0;
        outputRows.push([
          `${lo.toFixed(2)}-${hi.toFixed(2)}`,
          bTotal, bWins.length, r2(bWr), r2(bPnl), r4(bAvgPayout),
        ]);
      }
      outputRows.push([]);
    } catch(e) {
      console.log(`  ⚠️ Deep Edge: could not load ${s}: ${e.message}`);
    }
  }

  // Title
  const header = [`📊 Deep Edge Analysis — All ${stratCount} Strategies (WIN+LOSS only)`];
  const allRows = [header, [], ...outputRows];

  await sheets.spreadsheets.values.clear({ spreadsheetId: SSID, range: `'${tab}'!A1:G300` });
  await sheets.spreadsheets.values.update({
    spreadsheetId: SSID, range: `'${tab}'!A1`,
    valueInputOption: 'USER_ENTERED',
    requestBody: { values: allRows },
  });

  console.log(`  ✅ Deep Edge refreshed: ${stratCount} strategies`);
}


// ═══════════════════════════════════════════════════════════════════════════
// 👥 Crowd Accuracy — analyze crowd vs actual direction for all rounds
// ═══════════════════════════════════════════════════════════════════════════

async function refreshCrowdAccuracy(sheets, existingTabs) {
  const SSID = SPREADSHEET_ID;
  const tabName = '👥 Crowd Accuracy';

  // Load all_rounds data from the sheet tab
  const allRoundsTab = existingTabs.find(t => t.toLowerCase() === 'all_rounds');
  if (!allRoundsTab) {
    console.log('  ⏸ Crowd Accuracy: no all_rounds tab found — skipping');
    return;
  }

  let rows;
  try {
    const res = await sheets.spreadsheets.values.get({
      spreadsheetId: SSID,
      range: `'${allRoundsTab}'!A1:N50000`,
    });
    rows = res.data.values || [];
  } catch (e) {
    console.log(`  ⚠️ Crowd Accuracy: could not load all_rounds: ${e.message}`);
    return;
  }

  if (rows.length < 2) {
    console.log('  ⏸ Crowd Accuracy: all_rounds has no data');
    return;
  }

  const h = rows[0];
  const iEpoch = h.indexOf('epoch');
  const iLockTs = h.indexOf('lock_ts');
  const iTotalBnb = h.indexOf('final_total_bnb');
  const iBullPct = h.indexOf('final_bull_pct');
  const iBearPct = h.indexOf('final_bear_pct');
  const iConviction = h.indexOf('crowd_conviction_pct');
  const iCrowdSide = h.indexOf('crowd_side');
  const iActual = h.indexOf('actual_direction');
  const iCorrect = h.indexOf('crowd_correct');

  if (iEpoch < 0 || iCorrect < 0) {
    console.log('  ⏸ Crowd Accuracy: missing expected columns in all_rounds');
    return;
  }

  // Parse all rounds (skip FLAT outcomes)
  const rounds = [];
  for (let i = 1; i < rows.length; i++) {
    const r = rows[i];
    const actual = r[iActual] || '';
    if (actual === 'FLAT' || actual === '') continue;
    const correct = (r[iCorrect] || '').toLowerCase();
    rounds.push({
      epoch: parseInt(r[iEpoch]) || 0,
      lock_ts: parseFloat(r[iLockTs]) || 0,
      total_bnb: parseFloat(r[iTotalBnb]) || 0,
      bull_pct: parseFloat(r[iBullPct]) || 0,
      bear_pct: parseFloat(r[iBearPct]) || 0,
      conviction: parseFloat(r[iConviction]) || 0,
      crowd_side: r[iCrowdSide] || '',
      actual,
      correct: correct === 'true',
    });
  }

  if (rounds.length === 0) {
    console.log('  ⏸ Crowd Accuracy: no valid rounds to analyze');
    return;
  }

  const total = rounds.length;
  const correctCount = rounds.filter(r => r.correct).length;
  const wrongCount = total - correctCount;
  const accuracy = total > 0 ? correctCount / total * 100 : 0;

  // Section 2: Accuracy by conviction bucket
  const convBuckets = [
    { label: '50-55%', lo: 0.50, hi: 0.55 },
    { label: '55-60%', lo: 0.55, hi: 0.60 },
    { label: '60-70%', lo: 0.60, hi: 0.70 },
    { label: '70-80%', lo: 0.70, hi: 0.80 },
    { label: '80-90%', lo: 0.80, hi: 0.90 },
    { label: '90%+',   lo: 0.90, hi: 1.01 },
  ];

  const convRows = [
    ['Conviction Range', 'Rounds', 'Crowd Correct', 'Crowd Wrong', 'Accuracy %', 'Avg Pool Size (BNB)'],
  ];
  for (const b of convBuckets) {
    const bucket = rounds.filter(r => r.conviction >= b.lo && r.conviction < b.hi);
    const bCorrect = bucket.filter(r => r.correct).length;
    const bWrong = bucket.length - bCorrect;
    const bAcc = bucket.length > 0 ? r2(bCorrect / bucket.length * 100) : 0;
    const avgPool = bucket.length > 0 ? r2(bucket.reduce((s, r) => s + r.total_bnb, 0) / bucket.length) : 0;
    convRows.push([b.label, bucket.length, bCorrect, bWrong, bAcc, avgPool]);
  }

  // Section 3: Accuracy by pool size bucket
  const poolBuckets = [];
  poolBuckets.push({ label: '< 0.5 BNB', lo: 0, hi: 0.5 });
  for (let lo = 0.5; lo < 10; lo += 0.5) {
    const hi = lo + 0.5;
    poolBuckets.push({ label: `${lo}-${hi} BNB`, lo, hi });
  }
  poolBuckets.push({ label: '10+ BNB', lo: 10, hi: 1e9 });

  const poolRows = [
    ['Pool Size', 'Rounds', 'Accuracy %'],
  ];
  for (const b of poolBuckets) {
    const bucket = rounds.filter(r => r.total_bnb >= b.lo && r.total_bnb < b.hi);
    const bCorrect = bucket.filter(r => r.correct).length;
    const bAcc = bucket.length > 0 ? r2(bCorrect / bucket.length * 100) : 0;
    poolRows.push([b.label, bucket.length, bAcc]);
  }

  // Section 4: Hourly pattern (UTC)
  const hourlyData = {};
  for (let h = 0; h < 24; h++) hourlyData[h] = { rounds: [], convictionSum: 0 };
  for (const r of rounds) {
    if (r.lock_ts > 0) {
      const d = new Date(r.lock_ts * 1000);
      const hour = d.getUTCHours();
      hourlyData[hour].rounds.push(r);
      hourlyData[hour].convictionSum += r.conviction;
    }
  }

  const hourlyRows = [
    ['Hour (UTC)', 'Rounds', 'Accuracy %', 'Avg Conviction %'],
  ];
  for (let h = 0; h < 24; h++) {
    const bucket = hourlyData[h].rounds;
    const bCorrect = bucket.filter(r => r.correct).length;
    const bAcc = bucket.length > 0 ? r2(bCorrect / bucket.length * 100) : 0;
    const avgConv = bucket.length > 0 ? r2(hourlyData[h].convictionSum / bucket.length * 100) : 0;
    hourlyRows.push([String(h).padStart(2, '0'), bucket.length, bAcc, avgConv]);
  }

  // ── Section 5: Pool Activity Before Lock ──────────────────────────────
  // Load pool_snapshots tab if available
  const poolSnapshotsTab = existingTabs.find(t => t.toLowerCase() === 'pool_snapshots');
  let poolActivityRows = [['── Pool Activity Before Lock ──']];

  if (poolSnapshotsTab) {
    try {
      const psRes = await sheets.spreadsheets.values.get({
        spreadsheetId: SSID,
        range: `'${poolSnapshotsTab}'!A1:G200000`,
      });
      const psRows = psRes.data.values || [];

      if (psRows.length >= 2) {
        const psH = psRows[0];
        const psIEpoch = psH.indexOf('epoch');
        const psIStl = psH.indexOf('seconds_to_lock');
        const psITotal = psH.indexOf('total_bnb');

        if (psIEpoch >= 0 && psIStl >= 0 && psITotal >= 0) {
          // Parse all snapshots
          const allSnaps = [];
          for (let i = 1; i < psRows.length; i++) {
            const r = psRows[i];
            const epoch = parseInt(r[psIEpoch]) || 0;
            const stl = parseFloat(r[psIStl]) || 0;
            const total = parseFloat(r[psITotal]) || 0;
            if (epoch > 0 && stl > 0 && total > 0) {
              allSnaps.push({ epoch, stl, total });
            }
          }

          // Group by epoch
          const epochSnaps = {};
          for (const s of allSnaps) {
            if (!epochSnaps[s.epoch]) epochSnaps[s.epoch] = [];
            epochSnaps[s.epoch].push(s);
          }
          const epochList = Object.keys(epochSnaps).map(Number);

          // Time buckets to analyze
          const timeBuckets = [20, 15, 10, 7, 5, 4, 3, 2, 1];

          // Helper: find closest snapshot to target_stl within ±0.5s
          function findClosest(snaps, targetStl) {
            let best = null, bestDist = Infinity;
            for (const s of snaps) {
              const d = Math.abs(s.stl - targetStl);
              if (d <= 0.5 && d < bestDist) { best = s; bestDist = d; }
            }
            return best;
          }

          // Compute T-20s baseline per epoch
          const t20PerEpoch = {};
          for (const epoch of epochList) {
            const snap = findClosest(epochSnaps[epoch], 20);
            if (snap) t20PerEpoch[epoch] = snap.total;
          }

          const bucketStats = [];
          for (const tb of timeBuckets) {
            const poolValues = [];
            let stillGrowing = 0, epochsWithData = 0;

            for (const epoch of epochList) {
              const snap = findClosest(epochSnaps[epoch], tb);
              if (!snap) continue;
              poolValues.push(snap.total);

              // "Still growing" = pool at tb > pool at (tb+1)
              const nextBucketIdx = timeBuckets.indexOf(tb);
              const nextTb = nextBucketIdx > 0 ? timeBuckets[nextBucketIdx - 1] : tb + 1;
              const snapNext = findClosest(epochSnaps[epoch], nextTb);
              epochsWithData++;
              if (snapNext && snap.total > snapNext.total) stillGrowing++;
            }

            const avgPool = poolValues.length > 0
              ? r2(poolValues.reduce((a, b) => a + b, 0) / poolValues.length)
              : null;

            // Growth vs T-20s baseline
            const t20Vals = epochList
              .filter(e => t20PerEpoch[e])
              .map(e => {
                const snap = findClosest(epochSnaps[e], tb);
                return snap && t20PerEpoch[e] ? { pool: snap.total, base: t20PerEpoch[e] } : null;
              })
              .filter(Boolean);

            const avgGrowthPct = t20Vals.length > 0
              ? r2(t20Vals.reduce((s, v) => s + (v.pool - v.base) / v.base * 100, 0) / t20Vals.length)
              : null;

            const pctStillGrowing = epochsWithData > 0
              ? r2(stillGrowing / epochsWithData * 100)
              : null;

            bucketStats.push({
              tb,
              avgPool,
              avgGrowthPct,
              pctStillGrowing,
              n: poolValues.length,
            });
          }

          poolActivityRows.push([
            'Seconds Before Lock', 'Epochs w/ Data', 'Avg Pool Size (BNB)',
            'Avg Growth vs T-20s', '% Epochs Still Growing',
          ]);
          for (const b of bucketStats) {
            const growthStr = b.tb === 20
              ? 'baseline'
              : (b.avgGrowthPct !== null ? (b.avgGrowthPct >= 0 ? '+' : '') + b.avgGrowthPct + '%' : 'N/A');
            const growingStr = b.tb === 20 ? '-' : (b.pctStillGrowing !== null ? b.pctStillGrowing + '%' : 'N/A');
            poolActivityRows.push([
              b.tb + 's',
              b.n,
              b.avgPool !== null ? b.avgPool : 'N/A',
              growthStr,
              growingStr,
            ]);
          }
        } else {
          poolActivityRows.push(['Missing expected columns in pool_snapshots tab']);
        }
      } else {
        poolActivityRows.push(['No data in pool_snapshots tab yet']);
      }
    } catch (e) {
      poolActivityRows.push([`Error loading pool_snapshots: ${e.message}`]);
    }
  } else {
    poolActivityRows.push(['pool_snapshots tab not yet available']);
  }

  // Build output rows
  const outputRows = [
    ['👥 CROWD ACCURACY ANALYSIS'],
    [`Total rounds analyzed: ${total}`],
    [`Crowd correct: ${r2(accuracy)}%`],
    [`Crowd wrong: ${r2(100 - accuracy)}%`],
    [],
    ['── By Conviction ──────────────────────────'],
    ...convRows,
    [],
    ['── By Pool Size ────────────────────────────'],
    ...poolRows,
    [],
    ...poolActivityRows,
  ];

  // Hourly goes to the right (column H)
  const hourlyColRows = [['── Hourly Pattern (UTC) ────────────────────'], ...hourlyRows];

  // Create tab if needed
  const tabExists = existingTabs.some(t => t.toLowerCase() === tabName.toLowerCase());
  if (!tabExists) {
    await sheets.spreadsheets.batchUpdate({
      spreadsheetId: SSID,
      requestBody: { requests: [{ addSheet: { properties: { title: tabName } } }] },
    });
    existingTabs.push(tabName);
    console.log(`  📄 Created tab: ${tabName}`);
  }

  const actualTab = existingTabs.find(t => t.toLowerCase() === tabName.toLowerCase()) || tabName;

  // Clear and write
  await sheets.spreadsheets.values.clear({
    spreadsheetId: SSID,
    range: `'${actualTab}'!A1:M100`,
  });

  await sheets.spreadsheets.values.update({
    spreadsheetId: SSID,
    range: `'${actualTab}'!A1`,
    valueInputOption: 'USER_ENTERED',
    requestBody: { values: outputRows },
  });

  await sleep(1000);

  await sheets.spreadsheets.values.update({
    spreadsheetId: SSID,
    range: `'${actualTab}'!H1`,
    valueInputOption: 'USER_ENTERED',
    requestBody: { values: hourlyColRows },
  });

  console.log(`  ✅ Crowd Accuracy refreshed: ${total} rounds, accuracy=${r2(accuracy)}%`);
}


// ── Helpers ──────────────────────────────────────────────────────────────
function r2(n) { return Math.round(n * 100) / 100; }
function r4(n) { return Math.round(n * 10000) / 10000; }
function sleep(ms) { return new Promise(r => setTimeout(r, ms)); }

async function refreshEpochMap(sheets, existingTabs) {
  const SSID = SPREADSHEET_ID;
  // Dynamically detect strategy tabs (exclude analysis tabs)
  const analysisTabs = ['🏆 Strategy Comparison', '🔗 Cross-Strategy (Epoch)', '📊 Deep Edge Analysis', '🎯 Optimal Filters', '📈 PnL Curves', '🕐 Time Heatmap', '👥 Crowd Accuracy'];
  const STRATS = existingTabs.filter(t => !analysisTabs.includes(t) && !t.startsWith('combined_') && t !== 'all_rounds' && t !== 'pool_snapshots');
  const stratTab = existingTabs.find(t => t.toLowerCase() === '🔗 cross-strategy (epoch)'.toLowerCase())
    || '🔗 Cross-Strategy (Epoch)';

  if (!existingTabs.some(t => t.toLowerCase().includes('cross-strategy'))) {
    console.log('  ⏸ Cross-Strategy tab not found, skipping epoch map refresh');
    return;
  }

  // ── Load resolved trades per strategy ─────────────────────────────────
  const stratData = {};   // { stratName: { epoch: { side, open, close, edge, pnl, drift } } }
  const stratHasDrift = {};  // { stratName: boolean }
  const allEpochs = new Set();
  for (const s of STRATS) {
    const actualTab = existingTabs.find(t => t.toLowerCase() === s.toLowerCase()) || s;
    try {
      const res = await sheets.spreadsheets.values.get({ spreadsheetId: SSID, range: `'${actualTab}'!A1:AA5000` });
      const rows = res.data.values || [];
      if (rows.length < 2) continue;
      const headers = rows[0];
      const epochIdx = headers.indexOf('epoch');
      const sideIdx = headers.indexOf('side_label');
      const outcomeIdx = headers.indexOf('outcome');
      const openIdx = headers.indexOf('bnb_open');
      const closeIdx = headers.indexOf('bnb_close');
      const edgeIdx = headers.indexOf('edge_at_entry');
      const pnlIdx = headers.indexOf('pnl_usdc');
      const driftIdx = headers.indexOf('pool_drift_pct');
      stratHasDrift[s] = driftIdx >= 0;
      stratData[s] = {};
      for (let i = 1; i < rows.length; i++) {
        const r = rows[i];
        const epoch = r[epochIdx];
        if (!epoch) continue;
        const outcome = r[outcomeIdx] || '';
        if (outcome !== 'WIN' && outcome !== 'LOSS') continue;
        allEpochs.add(epoch);
        stratData[s][epoch] = {
          side: r[sideIdx] || '',
          outcome,
          open: parseFloat(r[openIdx]) || 0,
          close: parseFloat(r[closeIdx]) || 0,
          edge: parseFloat(r[edgeIdx]) || 0,
          pnl: parseFloat(r[pnlIdx]) || 0,
          drift: driftIdx >= 0 ? (parseFloat(r[driftIdx]) || 0) : null,
        };
      }
    } catch(e) {
      console.log(`  ⚠️ Could not load ${s}: ${e.message}`);
    }
  }

  const sortedEpochs = [...allEpochs].sort((a,b) => parseInt(a)-parseInt(b));
  const activeStrats = STRATS.filter(s => stratData[s] && Object.keys(stratData[s]).length > 0);

  // ── RIGHT SIDE: Epoch Map (J:...) ─────────────────────────────────────
  const driftCols = activeStrats.map(s => `${s}_drift`);
  const epRows = [
    ['═══ EPOCH MAP (refreshed hourly) ═══', ...Array(activeStrats.length + driftCols.length).fill('')],
    ['Epoch', ...activeStrats, 'Actual', 'UP votes', 'DOWN votes', ...driftCols],
  ];
  for (const epoch of sortedEpochs) {
    const sides = activeStrats.map(s => stratData[s]?.[epoch]?.side || '');
    let actual = '';
    for (const s of activeStrats) {
      const d = stratData[s]?.[epoch];
      if (d?.open > 0 && d?.close > 0) { actual = d.close > d.open ? 'UP' : 'DOWN'; break; }
    }
    const upVotes = sides.filter(s => s === 'UP').length;
    const downVotes = sides.filter(s => s === 'DOWN').length;
    const driftVals = activeStrats.map(s => {
      if (!stratHasDrift[s]) return '';
      const d = stratData[s]?.[epoch];
      if (!d || d.drift === null || d.drift === 0) return '';
      return r2(d.drift * 100);
    });
    epRows.push([epoch, ...sides, actual, upVotes, downVotes, ...driftVals]);
  }

  // ── LEFT SIDE: Per-Strategy Stats + Combinations (A:H) ───────────────
  const leftRows = [];

  // Header
  leftRows.push(['🔗 Cross-Strategy Consensus']);
  leftRows.push([]);
  leftRows.push([`📊 Per-Strategy Stats (${activeStrats.length} strategies, WIN+LOSS only)`]);
  leftRows.push(['Strategy', 'Trades', 'Wins', 'Win Rate %', 'PnL ($)', 'Avg Edge']);

  // Per-strategy stats (computed from data, not formulas)
  const stratStats = [];
  for (const s of activeStrats) {
    const trades = Object.values(stratData[s]);
    const wins = trades.filter(t => t.outcome === 'WIN').length;
    const losses = trades.filter(t => t.outcome === 'LOSS').length;
    const total = wins + losses;
    const wr = total > 0 ? (wins / total * 100) : 0;
    const pnl = trades.reduce((sum, t) => sum + t.pnl, 0);
    const avgEdge = total > 0 ? trades.reduce((sum, t) => sum + t.edge, 0) / total : 0;
    stratStats.push([
      s,
      total,
      wins,
      Math.round(wr * 100) / 100,
      Math.round(pnl * 100) / 100,
      Math.round(avgEdge * 10000) / 10000,
    ]);
  }
  // Sort by Win Rate % descending, strategies with < 30 trades at the bottom
  stratStats.sort((a, b) => {
    if (a[1] < 30 && b[1] >= 30) return 1;
    if (b[1] < 30 && a[1] >= 30) return -1;
    return b[3] - a[3];
  });
  for (const row of stratStats) {
    leftRows.push(row);
  }

  leftRows.push([]);

  // ── Combinations (pairs, triples, quadruples) ─────────────────────────
  // Only use strategies with at least 5 resolved trades for combinations
  const comboStrats = activeStrats.filter(s => Object.keys(stratData[s]).length >= 5);

  // Generate all pairs, triples, quadruples (hoisted for reuse in optimized block)
  const combos = [];
  const comboResults = [];

  if (comboStrats.length >= 2) {
    leftRows.push(['🏆 COMBINATIONS (Unanimous consensus)']);
    leftRows.push(['Combination', 'Size', 'Epochs', 'Wins', 'Win Rate %', 'PnL ($15/trade)', 'Better?']);

    const maxSize = Math.min(5, comboStrats.length);

    function generateCombos(start, current) {
      if (current.length >= 2 && current.length <= maxSize) {
        combos.push([...current]);
      }
      if (current.length >= maxSize) return;
      for (let i = start; i < comboStrats.length; i++) {
        current.push(comboStrats[i]);
        generateCombos(i + 1, current);
        current.pop();
      }
    }
    generateCombos(0, []);

    // Evaluate each combination
    for (const combo of combos) {
      let epochs = 0;
      let wins = 0;
      for (const epoch of sortedEpochs) {
        // Check if ALL strategies in combo voted on this epoch
        const votes = combo.map(s => stratData[s]?.[epoch]?.side).filter(Boolean);
        if (votes.length !== combo.length) continue;
        // Check unanimous
        const allUp = votes.every(v => v === 'UP');
        const allDown = votes.every(v => v === 'DOWN');
        if (!allUp && !allDown) continue;

        // Determine actual
        let actual = '';
        for (const s of activeStrats) {
          const d = stratData[s]?.[epoch];
          if (d?.open > 0 && d?.close > 0) { actual = d.close > d.open ? 'UP' : 'DOWN'; break; }
        }
        if (!actual) continue;

        epochs++;
        const consensusSide = allUp ? 'UP' : 'DOWN';
        if (consensusSide === actual) wins++;
      }

      const wr = epochs > 0 ? (wins / epochs * 100) : 0;
      const pnl = wins * 15 - (epochs - wins) * 15;
      const better = wr > 52 ? '✅ YES' : (wr >= 48 ? '➖ ~50/50' : '❌ NO');

      comboResults.push({
        name: combo.join('+'),
        size: combo.length,
        epochs,
        wins,
        wr: Math.round(wr * 100) / 100,
        pnl: Math.round(pnl * 100) / 100,
        better,
      });
    }

    // Sort: best win rate first (min 3 epochs to be meaningful)
    const sortedComboResults = [...comboResults].sort((a, b) => {
      if (a.epochs < 30 && b.epochs >= 30) return 1;
      if (b.epochs < 30 && a.epochs >= 30) return -1;
      return b.wr - a.wr;
    });

    for (const c of sortedComboResults) {
      leftRows.push([c.name, c.size, c.epochs, c.wins, c.wr, c.pnl, c.better]);
    }
  }

  // ── Compute profitable edge buckets per strategy ─────────────────────
  // Buckets: [0-0.05], [0.05-0.10], [0.10-0.15], [0.15-0.20], [0.20-0.30], [0.30+]
  const edgeBuckets = [[0, 0.05], [0.05, 0.10], [0.10, 0.15], [0.15, 0.20], [0.20, 0.30], [0.30, 1.0]];
  function bucketLabel(lo, hi) {
    return hi >= 1.0 ? `${lo.toFixed(2)}+` : `${lo.toFixed(2)}-${hi.toFixed(2)}`;
  }
  function stratAbbrev(s) {
    return s.split('_').map(w => w[0] ? w[0].toUpperCase() : '').join('');
  }

  const profitableBuckets = {}; // { stratName: [[lo, hi], ...] }
  for (const s of activeStrats) {
    const trades = Object.values(stratData[s]);
    profitableBuckets[s] = [];
    for (const [lo, hi] of edgeBuckets) {
      const bucket = trades.filter(t => t.edge >= lo && t.edge < hi);
      const bWins = bucket.filter(t => t.outcome === 'WIN').length;
      if (bucket.length >= 5 && bWins / bucket.length > 0.5) {
        profitableBuckets[s].push([lo, hi]);
      }
    }
  }

  // ── 📊 Consensus by count + Portfolio aggregate (I column, top) ─────
  const midRows = [];

  // Block 1: Consensus WR by number of agreeing strategies
  midRows.push(['📊 CONSENSUS BY COUNT']);
  midRows.push(['Min Strategies Agreeing', 'Epochs', 'Wins', 'Win Rate %', 'PnL ($10/trade)']);

  for (let minAgree = 2; minAgree <= activeStrats.length; minAgree++) {
    let epochs = 0, wins = 0;
    for (const epoch of sortedEpochs) {
      const sides = activeStrats.map(s => stratData[s]?.[epoch]?.side).filter(Boolean);
      if (sides.length < minAgree) continue;
      const upCount = sides.filter(s => s === 'UP').length;
      const downCount = sides.filter(s => s === 'DOWN').length;
      const maxAgree = Math.max(upCount, downCount);
      if (maxAgree < minAgree) continue;

      let actual = '';
      for (const s of activeStrats) {
        const d = stratData[s]?.[epoch];
        if (d?.open > 0 && d?.close > 0) { actual = d.close > d.open ? 'UP' : 'DOWN'; break; }
      }
      if (!actual) continue;

      epochs++;
      const consensusSide = upCount >= downCount ? 'UP' : 'DOWN';
      if (consensusSide === actual) wins++;
    }
    const wr = epochs > 0 ? r2(wins / epochs * 100) : 0;
    const pnl = r2(wins * 10 - (epochs - wins) * 10);
    midRows.push([`${minAgree}+ strategies agree`, epochs, wins, wr, pnl]);
  }

  // Pad midRows to exactly 17 rows so 🎯 starts at row 18 (I18)
  while (midRows.length < 19) {
    midRows.push([]);
  }

  // ── 🎯 Optimized Combinations — starts at I18 ──
  midRows.push(['🎯 OPTIMIZED COMBINATIONS (edge-filtered consensus)']);
  midRows.push(['Combination', 'Size', 'Epochs', 'Wins', 'Win Rate %', 'PnL ($15/trade)', 'Better?', 'Brute WR%', 'Edge Filter Used']);

  // ── 📈 Portfolio aggregate (separate block at O1) ──
  const portfolioRows = [];
  portfolioRows.push(['📈 PORTFOLIO AGGREGATE (all strategies independent)']);
  portfolioRows.push(['Metric', 'Value']);

  let totalTrades = 0, totalWins = 0, totalLosses = 0, totalPnl = 0;
  for (const s of activeStrats) {
    const trades = Object.values(stratData[s]);
    const w = trades.filter(t => t.outcome === 'WIN').length;
    const l = trades.filter(t => t.outcome === 'LOSS').length;
    const pnl = trades.reduce((sum, t) => sum + t.pnl, 0);
    totalTrades += w + l;
    totalWins += w;
    totalLosses += l;
    totalPnl += pnl;
  }

  const portfolioWr = totalTrades > 0 ? r2(totalWins / totalTrades * 100) : 0;
  const portfolioRoi = totalTrades > 0 ? r2(totalPnl / (totalTrades * 10) * 100) : 0;

  portfolioRows.push(['Total Trades', totalTrades]);
  portfolioRows.push(['Total Wins', totalWins]);
  portfolioRows.push(['Total Losses', totalLosses]);
  portfolioRows.push(['Win Rate %', portfolioWr]);
  portfolioRows.push(['Total PnL ($)', r2(totalPnl)]);
  portfolioRows.push(['ROI %', portfolioRoi]);
  portfolioRows.push(['Strategies Active', activeStrats.length]);

  const optEligibleStrats = comboStrats.filter(s => profitableBuckets[s] && profitableBuckets[s].length > 0);

  const optComboResults = [];
  for (const combo of combos) {
    // All strategies in combo must have profitable buckets
    if (!combo.every(s => optEligibleStrats.includes(s))) continue;

    let optEpochs = 0, optWins = 0;

    for (const epoch of sortedEpochs) {
      const votes = combo.map(s => stratData[s]?.[epoch]?.side).filter(Boolean);
      if (votes.length !== combo.length) continue;
      const allUp = votes.every(v => v === 'UP');
      const allDown = votes.every(v => v === 'DOWN');
      if (!allUp && !allDown) continue;

      // Check each strategy's edge was in one of its profitable buckets
      const allInProfitable = combo.every(s => {
        const edge = stratData[s]?.[epoch]?.edge;
        if (edge === undefined || edge === null) return false;
        return profitableBuckets[s].some(([lo, hi]) => edge >= lo && edge < hi);
      });
      if (!allInProfitable) continue;

      let actual = '';
      for (const s of activeStrats) {
        const d = stratData[s]?.[epoch];
        if (d?.open > 0 && d?.close > 0) { actual = d.close > d.open ? 'UP' : 'DOWN'; break; }
      }
      if (!actual) continue;

      optEpochs++;
      if ((allUp ? 'UP' : 'DOWN') === actual) optWins++;
    }

    if (optEpochs < 5) continue; // min 5 epochs (accumulation phase)

    const optWr = r2(optEpochs > 0 ? (optWins / optEpochs * 100) : 0);
    const optPnl = r2(optWins * 15 - (optEpochs - optWins) * 15);

    // Brute WR from comboResults
    const brute = comboResults.find(c => c.name === combo.join('+'));
    const bruteWr = brute ? brute.wr : 'N/A';

    // Build filter description
    const filterUsed = combo.map(s => {
      const labels = profitableBuckets[s].map(([lo, hi]) => bucketLabel(lo, hi)).join(',');
      return `${stratAbbrev(s)}:[${labels}]`;
    }).join(' ');

    const optBetter = optWr > 52 ? '✅ YES' : (optWr >= 48 ? '➖ ~50/50' : '❌ NO');
    optComboResults.push({ name: combo.join('+'), size: combo.length, epochs: optEpochs, wins: optWins, wr: optWr, pnl: optPnl, better: optBetter, bruteWr, filterUsed });
  }

  // Sort by WR descending
    optComboResults.sort((a, b) => {
    if (a.epochs < 30 && b.epochs >= 30) return 1;
    if (b.epochs < 30 && a.epochs >= 30) return -1;
    return b.wr - a.wr;
  });

  if (optComboResults.length === 0) {
    midRows.push(['Not enough data for optimized combinations']);
  } else {
    for (const c of optComboResults) {
      midRows.push([c.name, c.size, c.epochs, c.wins, c.wr, c.pnl, c.better, c.bruteWr, c.filterUsed]);
    }
  }

  // ── Write everything ──────────────────────────────────────────────────
  // Clear both sides
  await sheets.spreadsheets.values.clear({
    spreadsheetId: SSID,
    range: `'${stratTab}'!A1:H500`,
  });
  await sheets.spreadsheets.values.clear({
    spreadsheetId: SSID,
    range: `'${stratTab}'!I1:AZ6000`,
  });

  // Write left side (stats + combos) — unchanged
  await sheets.spreadsheets.values.update({
    spreadsheetId: SSID,
    range: `'${stratTab}'!A1`,
    valueInputOption: 'USER_ENTERED',
    requestBody: { values: leftRows },
  });
  await sleep(1200);

  // Write consensus + optimized combos at I1
  await sheets.spreadsheets.values.update({
    spreadsheetId: SSID,
    range: `'${stratTab}'!I1`,
    valueInputOption: 'USER_ENTERED',
    requestBody: { values: midRows },
  });
  await sleep(1200);

  // Write portfolio aggregate at O1
  await sheets.spreadsheets.values.update({
    spreadsheetId: SSID,
    range: `'${stratTab}'!O1`,
    valueInputOption: 'USER_ENTERED',
    requestBody: { values: portfolioRows },
  });
  await sleep(1200);

  // Write epoch map starting at column T (shifted right)
  await sheets.spreadsheets.values.update({
    spreadsheetId: SSID,
    range: `'${stratTab}'!T1`,
    valueInputOption: 'USER_ENTERED',
    requestBody: { values: epRows },
  });

  console.log(`  ✅ Cross-Strategy refreshed: ${sortedEpochs.length} epochs, ${activeStrats.length} strategies, ${optComboResults.length} optimized combos`);
}

main().catch(e => {
  console.error('❌', e.message);
  process.exit(1);
});
