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
      valueInputOption: 'RAW',
      requestBody: { values: rows },
    });

    console.log(`  ✅ ${actualTab}: ${rows.length - 1} trades synced`);
  }

  console.log('Done.');
}

main().catch(e => {
  console.error('❌', e.message);
  process.exit(1);
});
