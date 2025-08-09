import 'dotenv/config';
import { REST, Routes, SlashCommandBuilder } from 'discord.js';

const TOKEN = process.env.BOT_TOKEN;
const CLIENT_ID = process.env.CLIENT_ID; // application id
const GUILD_ID = process.env.GUILD_ID; // optional for guild registration

if (!TOKEN || !CLIENT_ID) {
  console.error('BOT_TOKEN and CLIENT_ID are required');
  process.exit(1);
}

const commands = [
  new SlashCommandBuilder()
    .setName('persist')
    .setDescription('Manage threads that should not auto-archive')
    .addSubcommand((sc) => sc
      .setName('add')
      .setDescription('Add this thread (or specified by thread_id) to watchlist')
      .addStringOption(o => o.setName('thread_id').setDescription('Optional thread ID').setRequired(false))
    )
    .addSubcommand((sc) => sc
      .setName('remove')
      .setDescription('Remove this thread (or specified by thread_id) from watchlist')
      .addStringOption(o => o.setName('thread_id').setDescription('Optional thread ID').setRequired(false))
    )
    .addSubcommand((sc) => sc
      .setName('list')
      .setDescription('List watched threads in this server')
    )
    .addSubcommand((sc) => sc
      .setName('keepalive')
      .setDescription('Trigger an immediate keepalive on this thread (or specified by thread_id)')
      .addStringOption(o => o.setName('thread_id').setDescription('Optional thread ID').setRequired(false))
    )
    .toJSON()
];

const rest = new REST({ version: '10' }).setToken(TOKEN);

async function main() {
  try {
    if (GUILD_ID) {
      await rest.put(Routes.applicationGuildCommands(CLIENT_ID, GUILD_ID), { body: commands });
      console.log('Registered guild commands');
    } else {
      await rest.put(Routes.applicationCommands(CLIENT_ID), { body: commands });
      console.log('Registered global commands');
    }
  } catch (err) {
    console.error('Failed to register commands', err);
    process.exit(1);
  }
}

main();