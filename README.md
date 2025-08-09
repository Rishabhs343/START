# Discord Thread Keepalive Bot

Prevents selected Discord threads from auto-archiving by periodically posting (and deleting) a tiny keepalive message. Includes slash commands to manage the watchlist.

## Why
Discord enforces auto-archive for threads after a max of 7 days of inactivity. This bot keeps specific threads alive by adding periodic activity.

## Features
- Slash commands: `/persist add`, `/persist remove`, `/persist list`, `/persist keepalive`
- Supports public, private, and announcement threads
- Unarchives threads as needed (requires Manage Threads)
- Sets the longest allowed auto-archive duration (1 week, with fallbacks)
- Sends a zero-width character and then deletes it to avoid clutter
- Schedules per-thread based on each thread's auto-archive duration, with jitter and a safety fraction
- JSON persistence in `data/persistent_threads.json`
- Health check endpoint at `/healthz`

## Permissions
When inviting the bot, grant at minimum:
- Send Messages
- Send Messages in Threads
- Manage Threads
- Read Message History

If you manage roles manually, ensure the bot role has these permissions in each relevant channel.

## Setup
1. Create a Discord Application and Bot in the Developer Portal.
2. Enable required intents (only default intents are needed here).
3. Create `.env` from `.env.example` and fill `BOT_TOKEN` (and `CLIENT_ID` if you want to run `npm run register`).
4. Install and run:
   ```bash
   npm install
   npm run start
   ```
5. Invite the bot to your server with a link that includes the needed permissions.

### Slash Commands
Commands are auto-registered per-guild when the bot starts. If you prefer explicit registration, set `CLIENT_ID` (and optionally `GUILD_ID`) and run:
```bash
npm run register
```
- Guild registration is instant but per server.
- Global registration can take up to an hour to propagate.

### Usage
- In a thread, run:
  ```
  /persist add
  ```
  This adds the current thread to the watchlist and schedules keepalives.
- To stop keeping a thread alive:
  ```
  /persist remove
  ```
- To list watched threads in the current server:
  ```
  /persist list
  ```
- To manually bump now:
  ```
  /persist keepalive
  ```

## Configuration
- `KEEPALIVE_INTERVAL_HOURS` (default 120 = 5 days): Upper cap for base interval
- `SCHEDULE_FRACTION` (default 0.7): Base interval is min(KEEPALIVE_INTERVAL_HOURS, auto-archive-hours × SCHEDULE_FRACTION)
- `JITTER_HOURS` (default 12): Random spread added to the base interval
- `KEEPALIVE_DELETE_DELAY_MS` (default 2000): Delay before deleting the keepalive message
- `DATA_DIR` (default `./data`): Where the JSON store lives
- `PORT` (default 3000): Health check port

## Notes and Loopholes Covered
- Cannot disable auto-archive: bot unarchives and bumps as needed
- Threads older than 3 days: requires Manage Threads to unarchive — bot handles and retries
- Private threads: bot joins before sending
- Locked threads / missing perms: failures are tracked; after 10 consecutive failures the thread is removed from watchlist
- Server boost limits: bot attempts to set 1 week auto-archive; gracefully falls back to 3d/1d/1h
- Per-thread scheduling: respects each thread’s archive duration, reducing unnecessary bumps
- Rate limits: operations are staggered with jitter; minimal API calls per cycle
- Cleanup: keepalive message is deleted to avoid clutter
- Persistence: JSON store survives restarts and reschedules appropriately

## Docker
```dockerfile
# Buildless Node image
FROM node:20-alpine
WORKDIR /app
COPY package.json package-lock.json* ./
RUN npm install --omit=dev
COPY . .
CMD ["npm", "start"]
```
Build and run:
```bash
docker build -t thread-keepalive-bot .
docker run -e BOT_TOKEN=... -p 3000:3000 -v $(pwd)/data:/app/data thread-keepalive-bot
```

## Troubleshooting
- 403 when sending or unarchiving: Check bot permissions on the thread’s parent channel
- Thread not found: Might be deleted — it will be removed from the watchlist automatically
- Commands not visible: Ensure the bot has application commands permission and that registration completed (check logs) 
