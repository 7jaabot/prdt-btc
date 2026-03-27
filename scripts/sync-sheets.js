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
        csvFiles.push({ path: csvPath, tabName, strategy, mode });
      }
    }
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

  // Refresh the Cross-Strategy Epoch Map
  await refreshEpochMap(sheets, existingTabs);

  console.log('Done.');
}

async function refreshEpochMap(sheets, existingTabs) {
  const SSID = SPREADSHEET_ID;
  const STRATS = ['GBM', 'follow_crowd', 'mean_reversion', 'orderbook', 'pool_contrarian', 'manual_direction'];
  const stratTab = existingTabs.find(t => t.toLowerCase() === '🔗 cross-strategy (epoch)'.toLowerCase())
    || '🔗 Cross-Strategy (Epoch)';

  if (!existingTabs.some(t => t.toLowerCase().includes('cross-strategy'))) {
    console.log('  ⏸ Cross-Strategy tab not found, skipping epoch map refresh');
    return;
  }

  // Load resolved trades per strategy
  const stratData = {};
  const allEpochs = new Set();
  for (const s of STRATS) {
    const actualTab = existingTabs.find(t => t.toLowerCase() === s.toLowerCase()) || s;
    try {
      const res = await sheets.spreadsheets.values.get({ spreadsheetId: SSID, range: `'${actualTab}'!A1:V500` });
      const rows = res.data.values || [];
      if (rows.length < 2) continue;
      const headers = rows[0];
      const epochIdx = headers.indexOf('epoch');
      const sideIdx = headers.indexOf('side_label');
      const outcomeIdx = headers.indexOf('outcome');
      const openIdx = headers.indexOf('bnb_open');
      const closeIdx = headers.indexOf('bnb_close');
      const edgeIdx = headers.indexOf('edge_at_entry');
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
          open: parseFloat(r[openIdx]) || 0,
          close: parseFloat(r[closeIdx]) || 0,
        };
      }
    } catch(e) {
      console.log(`  ⚠️ Could not load ${s}: ${e.message}`);
    }
  }

  const sortedEpochs = [...allEpochs].sort((a,b) => parseInt(a)-parseInt(b));

  // Build epoch map rows (starts at row 21 in the sheet — keep same structure)
  const epRows = [['Epoch', ...STRATS, 'Actual', 'UP votes', 'DOWN votes']];
  for (const epoch of sortedEpochs) {
    const sides = STRATS.map(s => stratData[s]?.[epoch]?.side || '');
    let actual = '';
    for (const s of STRATS) {
      const d = stratData[s]?.[epoch];
      if (d?.open > 0 && d?.close > 0) { actual = d.close > d.open ? 'UP' : 'DOWN'; break; }
    }
    const upVotes = sides.filter(s => s === 'UP').length;
    const downVotes = sides.filter(s => s === 'DOWN').length;
    epRows.push([epoch, ...sides, actual, upVotes, downVotes]);
  }

  // Clear and rewrite from row 20 (header) onward — leaves rows 1-19 (stats/filters) intact
  await sheets.spreadsheets.values.clear({
    spreadsheetId: SSID,
    range: `'${stratTab}'!A20:J600`,
  });
  await sheets.spreadsheets.values.update({
    spreadsheetId: SSID,
    range: `'${stratTab}'!A20`,
    valueInputOption: 'USER_ENTERED',
    requestBody: { values: epRows },
  });

  console.log(`  ✅ Epoch Map refreshed: ${sortedEpochs.length} epochs`);
}

main().catch(e => {
  console.error('❌', e.message);
  process.exit(1);
});
