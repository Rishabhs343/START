import 'dotenv/config';
import express from 'express';
import {
  ChannelType,
  Client,
  GatewayIntentBits,
  RESTJSONErrorCodes,
  SlashCommandBuilder,
} from 'discord.js';
import { readStore, upsertThread, removeThread, listThreads } from './store.js';

// ---- Configuration ----
const KEEPALIVE_INTERVAL_HOURS = Number(process.env.KEEPALIVE_INTERVAL_HOURS || 120); // default 5 days
const KEEPALIVE_DELETE_DELAY_MS = Number(process.env.KEEPALIVE_DELETE_DELAY_MS || 2000);
const JITTER_HOURS = Number(process.env.JITTER_HOURS || 12); // randomize to spread load
const SCHEDULE_FRACTION = Math.min(0.95, Math.max(0.2, Number(process.env.SCHEDULE_FRACTION || 0.7))); // 20%..95%
const PORT = Number(process.env.PORT || 3000);
const TOKEN = process.env.BOT_TOKEN;

if (!TOKEN) {
  console.error('BOT_TOKEN is required in environment');
  process.exit(1);
}

// ---- HTTP Healthcheck ----
const app = express();
app.get('/healthz', (_req, res) => res.json({ ok: true }));
app.get('/', (_req, res) => res.send('Discord Thread Keepalive Bot running'));
app.listen(PORT, () => console.log(`[health] listening on :${PORT}`));

// ---- Discord Client ----
const client = new Client({ intents: [GatewayIntentBits.Guilds, GatewayIntentBits.GuildMessages] });

// In-memory schedule handles
const threadIdToTimeout = new Map();
const threadIdToFailureCount = new Map();
const threadIdToBaseHours = new Map(); // per-thread base interval derived from auto-archive

function hoursToMs(hours) {
  return Math.max(1, Math.floor(hours * 60 * 60 * 1000));
}

function minutesToHours(mins) {
  return Math.max(0.0167, mins / 60);
}

function computeBaseHoursFromAutoArchiveMinutes(autoArchiveMinutes) {
  // Derive a safe base interval from the thread's auto-archive duration
  const archiveHours = minutesToHours(autoArchiveMinutes || 10080); // default 1w
  const derived = Math.max(0.5, archiveHours * SCHEDULE_FRACTION); // at least 30 minutes
  // Also respect a global cap if set lower than derived
  return Math.min(derived, KEEPALIVE_INTERVAL_HOURS);
}

function getNextDelayMs(threadId) {
  const baseHours = threadIdToBaseHours.get(threadId) ?? KEEPALIVE_INTERVAL_HOURS;
  const base = hoursToMs(baseHours);
  const jitter = Math.floor(Math.random() * hoursToMs(JITTER_HOURS));
  return base + jitter;
}

async function ensureThreadSettings(thread) {
  // Try to set the max auto-archive duration available. Fallbacks if disallowed
  const desiredDurations = [10080, 4320, 1440, 60]; // minutes: 1w, 3d, 1d, 1h
  for (const minutes of desiredDurations) {
    try {
      if (thread.autoArchiveDuration !== minutes) {
        await thread.setAutoArchiveDuration(minutes, 'ensure max auto-archive duration');
      }
      return thread.autoArchiveDuration ?? minutes; // succeeded
    } catch (err) {
      // Ignore and try next
    }
  }
  return thread.autoArchiveDuration ?? 1440; // fallback assume 1d
}

async function keepaliveThread(threadId) {
  try {
    const channel = await client.channels.fetch(threadId);
    if (!channel || (channel.type !== ChannelType.PublicThread && channel.type !== ChannelType.PrivateThread && channel.type !== ChannelType.AnnouncementThread)) {
      console.warn(`[keepalive] ${threadId}: not a thread or not found, removing`);
      removeThread(threadId);
      cancelSchedule(threadId);
      return;
    }
    const thread = channel;

    if (thread.archived) {
      try {
        await thread.setArchived(false, 'keepalive unarchive');
      } catch (err) {
        // If locked or insufficient perms
        console.warn(`[keepalive] ${threadId}: failed to unarchive (${err.code || err.message}), will retry later`);
        incrementFailure(threadId);
        return;
      }
    }

    // Join if not already a member (for private threads)
    if (typeof thread.join === 'function') {
      try {
        await thread.join();
      } catch (_) {
        // ignore
      }
    }

    const appliedMinutes = await ensureThreadSettings(thread);

    // Send and delete a zero-width space to bump activity without clutter
    let message;
    try {
      message = await thread.send('\u200B');
    } catch (err) {
      // If cannot send (locked, missing perms)
      console.warn(`[keepalive] ${threadId}: failed to send (${err.code || err.message})`);
      incrementFailure(threadId);
      return;
    }

    // Schedule deletion, but deletion is best-effort
    setTimeout(() => {
      message.delete().catch(() => {});
    }, KEEPALIVE_DELETE_DELAY_MS);

    // Success; reset failure count and update store timestamp
    threadIdToFailureCount.delete(threadId);
    upsertThread({ threadId, guildId: thread.guildId, lastKeepaliveAt: Date.now() });

    // Update per-thread base hours from current auto-archive duration
    const baseHours = computeBaseHoursFromAutoArchiveMinutes(appliedMinutes ?? thread.autoArchiveDuration);
    threadIdToBaseHours.set(threadId, baseHours);

    console.log(`[keepalive] ${threadId}: ok; base=${baseHours}h, autoArchive=${appliedMinutes}m`);
  } catch (err) {
    if (err && err.code === RESTJSONErrorCodes.UnknownChannel) {
      console.warn(`[keepalive] ${threadId}: channel deleted, removing`);
      removeThread(threadId);
      cancelSchedule(threadId);
      return;
    }
    console.warn(`[keepalive] ${threadId}: unexpected error`, err);
    incrementFailure(threadId);
  }
}

function incrementFailure(threadId) {
  const count = (threadIdToFailureCount.get(threadId) || 0) + 1;
  threadIdToFailureCount.set(threadId, count);
  // After many consecutive failures, remove from list (likely locked or perms)
  if (count >= 10) {
    console.warn(`[keepalive] ${threadId}: too many failures, removing from watchlist`);
    removeThread(threadId);
    cancelSchedule(threadId);
  }
}

function scheduleThread(threadId, initialDelayMs = null) {
  cancelSchedule(threadId);
  const delay = initialDelayMs ?? getNextDelayMs(threadId);
  const timeout = setTimeout(async () => {
    await keepaliveThread(threadId);
    scheduleThread(threadId); // reschedule with new randomized delay based on per-thread base
  }, delay);
  threadIdToTimeout.set(threadId, timeout);
  const hours = (delay / (1000 * 60 * 60)).toFixed(2);
  console.log(`[schedule] ${threadId}: next in ${hours}h`);
}

function cancelSchedule(threadId) {
  const existing = threadIdToTimeout.get(threadId);
  if (existing) {
    clearTimeout(existing);
    threadIdToTimeout.delete(threadId);
  }
}

async function primeThreadBaseHours(threadId) {
  try {
    const channel = await client.channels.fetch(threadId);
    if (!channel?.isThread?.()) return;
    const minutes = channel.autoArchiveDuration ?? 1440;
    const base = computeBaseHoursFromAutoArchiveMinutes(minutes);
    threadIdToBaseHours.set(threadId, base);
  } catch (_) {
    // ignore; will be set on first successful keepalive
  }
}

async function scheduleAllOnStartup() {
  const store = readStore();
  // Prime base hours for all threads in parallel
  await Promise.all(store.threads.map(t => primeThreadBaseHours(t.threadId)));
  for (const entry of store.threads) {
    // Stagger quick initial sweep to calibrate and bump if needed
    const staggerMinutes = 5 + Math.floor(Math.random() * 25); // 5..30 minutes
    scheduleThread(entry.threadId, staggerMinutes * 60 * 1000);
  }
}

// ---- Slash Commands ----
const commands = [
  new SlashCommandBuilder()
    .setName('persist')
    .setDescription('Manage threads that should not auto-archive')
    .addSubcommand((sc) => sc
      .setName('add')
      .setDescription('Add this thread (or a specified thread) to the keepalive watchlist')
      .addStringOption((opt) => opt
        .setName('thread_id')
        .setDescription('Optional: specify a thread ID; default is the current thread')
        .setRequired(false)
      )
    )
    .addSubcommand((sc) => sc
      .setName('remove')
      .setDescription('Remove this thread (or a specified thread) from the watchlist')
      .addStringOption((opt) => opt
        .setName('thread_id')
        .setDescription('Optional: specify a thread ID; default is the current thread')
        .setRequired(false)
      )
    )
    .addSubcommand((sc) => sc
      .setName('list')
      .setDescription('List watched threads in this server')
    )
    .addSubcommand((sc) => sc
      .setName('keepalive')
      .setDescription('Trigger an immediate keepalive on this thread (or specified thread)')
      .addStringOption((opt) => opt
        .setName('thread_id')
        .setDescription('Optional: specify a thread ID; default is the current thread')
        .setRequired(false)
      )
    )
    .toJSON()
];

client.on('ready', async () => {
  console.log(`Logged in as ${client.user.tag}`);
  // Register commands per guild for immediate availability
  for (const [guildId, guild] of client.guilds.cache) {
    try {
      await guild.commands.set(commands);
      console.log(`[commands] registered in guild ${guild.name} (${guildId})`);
    } catch (err) {
      console.warn(`[commands] failed to register in guild ${guildId}`, err);
    }
  }
  await scheduleAllOnStartup();
});

client.on('guildCreate', async (guild) => {
  try {
    await guild.commands.set(commands);
    console.log(`[commands] registered in new guild ${guild.name} (${guild.id})`);
  } catch (err) {
    console.warn(`[commands] failed to register in new guild ${guild.id}`, err);
  }
});

client.on('interactionCreate', async (interaction) => {
  if (!interaction.isChatInputCommand()) return;
  if (interaction.commandName !== 'persist') return;

  const sub = interaction.options.getSubcommand();
  const explicitThreadId = interaction.options.getString('thread_id') || null;
  const contextChannel = interaction.channel;
  const fallbackThreadId = contextChannel?.isThread?.() ? contextChannel.id : null;
  const targetThreadId = explicitThreadId || fallbackThreadId;

  if (sub !== 'list' && !targetThreadId) {
    await interaction.reply({ ephemeral: true, content: 'Run this inside a thread or provide a thread_id option.' });
    return;
  }

  try {
    if (sub === 'add') {
      const channel = await client.channels.fetch(targetThreadId);
      if (!channel?.isThread?.()) {
        await interaction.reply({ ephemeral: true, content: 'Target is not a thread or cannot be accessed.' });
        return;
      }
      // Prime per-thread base hours
      const minutes = channel.autoArchiveDuration ?? 1440;
      threadIdToBaseHours.set(channel.id, computeBaseHoursFromAutoArchiveMinutes(minutes));
      upsertThread({ threadId: channel.id, guildId: interaction.guildId, lastKeepaliveAt: 0 });
      // Schedule a quick initial keepalive to calibrate and ensure it's active
      scheduleThread(channel.id, 10 * 1000);
      await interaction.reply({ ephemeral: true, content: `Thread added to keepalive: ${channel.name} (${channel.id})` });
    } else if (sub === 'remove') {
      removeThread(targetThreadId);
      cancelSchedule(targetThreadId);
      await interaction.reply({ ephemeral: true, content: `Thread removed from keepalive: ${targetThreadId}` });
    } else if (sub === 'list') {
      const threads = listThreads().filter(t => t.guildId === interaction.guildId);
      if (threads.length === 0) {
        await interaction.reply({ ephemeral: true, content: 'No watched threads in this server.' });
        return;
      }
      const lines = threads.map(t => `• <#${t.threadId}> (id: ${t.threadId})`);
      await interaction.reply({ ephemeral: true, content: `Watched threads (server):\n${lines.join('\n')}` });
    } else if (sub === 'keepalive') {
      await interaction.deferReply({ ephemeral: true });
      await keepaliveThread(targetThreadId);
      await interaction.editReply({ content: `Keepalive attempted for thread ${targetThreadId}. Check logs for details.` });
    }
  } catch (err) {
    console.error('[interaction] error', err);
    if (!interaction.replied) {
      await interaction.reply({ ephemeral: true, content: 'An error occurred. Check bot logs.' });
    }
  }
});

client.login(TOKEN);