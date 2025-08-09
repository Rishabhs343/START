import fs from 'fs';
import path from 'path';

const DATA_DIR = process.env.DATA_DIR || path.resolve(process.cwd(), 'data');
const STORE_PATH = path.join(DATA_DIR, 'persistent_threads.json');

function ensureDataDir() {
  if (!fs.existsSync(DATA_DIR)) fs.mkdirSync(DATA_DIR, { recursive: true });
}

function initStore() {
  ensureDataDir();
  if (!fs.existsSync(STORE_PATH)) {
    const initial = { threads: [] };
    fs.writeFileSync(STORE_PATH, JSON.stringify(initial, null, 2));
  }
}

export function readStore() {
  initStore();
  try {
    const raw = fs.readFileSync(STORE_PATH, 'utf8');
    const parsed = JSON.parse(raw);
    if (!parsed.threads || !Array.isArray(parsed.threads)) return { threads: [] };
    return parsed;
  } catch (_) {
    return { threads: [] };
  }
}

let writeInFlight = false;
let pendingWrite = false;

function writeSafely(data) {
  if (writeInFlight) {
    pendingWrite = true;
    return;
  }
  writeInFlight = true;
  const tmp = `${STORE_PATH}.tmp`;
  fs.writeFile(tmp, JSON.stringify(data, null, 2), (err) => {
    if (err) {
      writeInFlight = false;
      if (pendingWrite) {
        pendingWrite = false;
        writeSafely(data);
      }
      return;
    }
    fs.rename(tmp, STORE_PATH, (renameErr) => {
      writeInFlight = false;
      if (pendingWrite) {
        pendingWrite = false;
        writeSafely(data);
      }
      if (renameErr) {
        // best-effort; leave tmp in place
      }
    });
  });
}

export function writeStore(next) {
  writeSafely(next);
}

export function upsertThread(entry) {
  const store = readStore();
  const idx = store.threads.findIndex(t => t.threadId === entry.threadId);
  if (idx === -1) {
    store.threads.push({ threadId: entry.threadId, guildId: entry.guildId, lastKeepaliveAt: entry.lastKeepaliveAt || 0 });
  } else {
    store.threads[idx] = { ...store.threads[idx], ...entry };
  }
  writeStore(store);
}

export function removeThread(threadId) {
  const store = readStore();
  const next = { threads: store.threads.filter(t => t.threadId !== threadId) };
  writeStore(next);
}

export function listThreads() {
  const store = readStore();
  return store.threads;
}