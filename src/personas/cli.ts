#!/usr/bin/env node
// `pnpm personas` — CLI for the v2 persona system.
//
// Subcommands:
//   --list                        List all personas
//   --create <seed-id>            Create a new persona from a seed (dry-run if --dry-run)
//   --status [<id>]               Show one or all personas
//   --validate <id>               Validate the session (browserless; falls back to browser)
//   --probe <id> "<prompt>"       Send a single prompt
//   --converse <id> <file>        Sequence of prompts
//   --batch <id> <file>           One prompt per line, fresh conv each
//   --dump <id>                   Print full persona state (sanitized)
//   --audit <id> [--limit N]      Tail the audit log
//   --archive <id>                Remove from index (files stay on disk)
//   --health                      Status of all personas
//   --init-keys                   Generate PERSONA_MASTER_KEY and print
//   --help                        Show usage

// Note: we intentionally do NOT import `../config.js` because that module
// requires OPENAI_ADS_API_KEY and VERSEODIN_API_KEY to be set. The personas
// CLI should work standalone (e.g. on a personal laptop without those keys).
import "dotenv/config";
import { readFile } from "node:fs/promises";
import { PersonaManager, makePersonaFromSeed, createWithBrowser } from "./manager.js";
import { ChatGPTClient } from "./chatgpt/client.js";
import { BrowserPersonaRunner } from "./browser/runner.js";
import { poolFromEnv } from "./proxy/pool.js";
import { ensureHome, identityExists, personaDir, readIdentity } from "./storage.js";
import { asPersonaId, PersonaId, AuthState } from "./types.js";
import { listPersonaIds, readEnrichedIndex } from "./registry.js";
import { getSeed, PERSONA_SEEDS } from "./seeds/index.js";

const USAGE = `
pnpm personas — manage persistent ChatGPT persona accounts

Usage:
  pnpm personas --list
  pnpm personas --create <seed-id> [--dry-run] [--tags t1,t2] [--id <new-id>]
  pnpm personas --create-account <seed-id>   [BROWSER] signs up a real account
  pnpm personas --status [<id>]
  pnpm personas --validate <id>
  pnpm personas --import-session <id> "<token>" [--cf "<cf_clearance>"]
  pnpm personas --probe <id> "<prompt>"
  pnpm personas --converse <id> <file>
  pnpm personas --batch <id> <file>
  pnpm personas --multi-batch <file> [--concurrency N]   one "persona|prompt" per line
  pnpm personas --dump <id>
  pnpm personas --audit <id> [--limit N]
  pnpm personas --archive <id>
  pnpm personas --refresh-cf <id>            [BROWSER] refreshes cf_clearance
  pnpm personas --reauth <id>                [BROWSER] full re-auth via email code
  pnpm personas --rotate-proxy <id>          rotate the persona's proxy session
  pnpm personas --proxy-info [<id>]          show pool + per-persona proxy config
  pnpm personas --proxy-test                verify egress IP via ip.oxylabs.io
  pnpm personas --health
  pnpm personas --init-keys

Notes:
  Commands marked [BROWSER] launch a Chromium with automation plugins.
  They will be flagged by EDR products. Only run them on a machine
  where this is acceptable (e.g. personal laptop, VPS).

  Day-to-day probing (--probe, --batch, --converse, --validate) is
  browserless and uses the configured proxy (OXYLABS_PROXY_USERNAME etc).

Proxy env vars (optional):
  OXYLABS_PROXY_USERNAME  — your Oxylabs customer username (no "customer-" prefix)
  OXYLABS_PROXY_PASSWORD  — your Oxylabs password
  OXYLABS_PROXY_TYPE      — isp (default) | residential | datacenter
  OXYLABS_PROXY_COUNTRY   — cc code, default US
  OXYLABS_PROXY_STATE     — e.g. us-ca
  OXYLABS_PROXY_CITY      — e.g. los_angeles
  OXYLABS_PROXY_TTL_MIN   — sticky session TTL, default 30
  PERSONA_PROXY_DISABLED  — "true" to disable proxy entirely (local egress)

Seeds available:
${PERSONA_SEEDS.map((s) => `  - ${s.id.padEnd(20)} ${s.label}`).join("\n")}
`;

function arg(name: string): string | undefined {
  const i = process.argv.indexOf(name);
  if (i < 0) return undefined;
  return process.argv[i + 1];
}
function flag(name: string): boolean {
  return process.argv.includes(name);
}

function masterKeyOrThrow(): string {
  const k = process.env.PERSONA_MASTER_KEY;
  if (!k || !/^[0-9a-fA-F]{64}$/.test(k)) {
    console.error(
      "PERSONA_MASTER_KEY is missing or invalid.\n" +
        "It must be 64 hex chars (32 bytes). Generate with:\n" +
        "  pnpm personas --init-keys",
    );
    process.exit(1);
  }
  return k;
}

async function main(): Promise<void> {
  if (process.argv.length === 2 || flag("--help") || flag("-h")) {
    console.log(USAGE);
    return;
  }

  if (flag("--init-keys")) {
    const { generateMasterKey } = await import("./crypto.js");
    const k = generateMasterKey();
    console.log(k);
    console.log("\n# add to .env:");
    console.log(`PERSONA_MASTER_KEY=${k}`);
    return;
  }

  await ensureHome();

  // --list / --status / --health don't need the master key (read-only on public files).
  if (flag("--list")) {
    return cmdList();
  }
  if (flag("--health")) {
    return cmdHealth();
  }
  const statusId = arg("--status");
  if (statusId !== undefined) {
    return cmdStatus(statusId === "true" || statusId === "*" ? undefined : asPersonaId(statusId));
  }

  // Everything below mutates secrets; require the master key.
  const masterKey = masterKeyOrThrow();
  const proxyPool = poolFromEnv();
  const pm = new PersonaManager({
    masterKeyHex: masterKey,
    home: process.env.OPENAI_ADS_HOME,
    proxyPool,
    clientFactory: (persona, auth, ctx) =>
      new ChatGPTClient(persona, auth, { proxyUrl: ctx.proxyUrl }),
    // Browser factory is lazy-instantiated; only loaded when actually used.
    browserFactory: (browserOpts) =>
      new BrowserPersonaRunner({
        headless: flag("--headed") ? false : true,
        proxy: browserOpts?.proxy,
      }),
  });
  await pm.init();

  if (flag("--create")) return cmdCreate(pm, arg("--create")!);
  if (flag("--create-account")) return cmdCreateAccount(pm, arg("--create-account")!);
  if (flag("--dump")) return cmdDump(pm, asPersonaId(arg("--dump")!));
  if (flag("--audit")) return cmdAudit(pm, asPersonaId(arg("--audit")!), arg("--limit"));
  if (flag("--archive")) return cmdArchive(pm, asPersonaId(arg("--archive")!));
  if (flag("--refresh-cf")) return cmdRefreshCf(pm, asPersonaId(arg("--refresh-cf")!));
  if (flag("--reauth")) return cmdReauth(pm, asPersonaId(arg("--reauth")!));
  if (flag("--rotate-proxy")) return cmdRotateProxy(pm, asPersonaId(arg("--rotate-proxy")!));
  if (flag("--proxy-info")) return cmdProxyInfo(pm, arg("--proxy-info"));
  if (flag("--proxy-test")) return cmdProxyTest(proxyPool);

  // Network-bearing commands
  if (flag("--validate")) return cmdValidate(pm, asPersonaId(arg("--validate")!));
  if (flag("--import-session")) {
    const id = asPersonaId(arg("--import-session")!);
    const token = positional(1);
    const cf = arg("--cf");
    if (!token) {
      console.error('Usage: pnpm personas --import-session <id> "<session_token>" [--cf "<cf_clearance>"]');
      process.exit(1);
    }
    return cmdImportSession(pm, id, token, cf);
  }
  if (flag("--probe")) {
    const id = asPersonaId(arg("--probe")!);
    const prompt = arg("--prompt") ?? positional(1);
    if (!prompt) {
      console.error('Usage: pnpm personas --probe <id> --prompt "<text>"');
      process.exit(1);
    }
    return cmdProbe(pm, id, prompt);
  }
  if (flag("--converse")) return cmdConverse(pm, asPersonaId(arg("--converse")!), arg("--converse-file") ?? positional(1));
  if (flag("--batch")) return cmdBatch(pm, asPersonaId(arg("--batch")!), arg("--batch-file") ?? positional(1));
  if (flag("--multi-batch")) {
    const { runMultiBatch } = await import("./multi-batch.js");
    const filePath = arg("--multi-batch-file") ?? positional(1) ?? "-";
    const concurrency = parseInt(arg("--concurrency") ?? "5", 10);
    return runMultiBatch(pm, filePath, { concurrency });
  }

  console.error("Unknown command. Try --help.");
  process.exit(1);
}

async function cmdList(): Promise<void> {
  const idx = await readEnrichedIndex();
  const ids = Object.keys(idx.personas).sort();
  if (ids.length === 0) {
    console.log("\nNo personas yet. Run: pnpm personas --create <seed-id>\n");
    return;
  }
  console.log("");
  console.log(
    pad("ID", 28) + pad("LABEL", 30) + pad("PLAN", 12) + pad("HEALTH", 16) + pad("SCORE", 6) + pad("PROBES", 8) + "LAST USED",
  );
  console.log("─".repeat(112));
  for (const id of ids) {
    const p = idx.personas[id]!;
    console.log(
      pad(p.id, 28) +
        pad(p.label, 30) +
        pad(p.plan, 12) +
        pad(p.health, 16) +
        pad(String(p.healthScore), 6) +
        pad(String(p.totalProbes), 8) +
        p.lastUsed,
    );
  }
  console.log("");
}

async function cmdStatus(id?: PersonaId): Promise<void> {
  if (!id) {
    const ids = await listPersonaIds();
    if (ids.length === 0) {
      console.log("No personas.");
      return;
    }
    for (const i of ids) {
      await cmdStatus(i);
    }
    return;
  }
  const persona = await readIdentity(id);
  console.log("");
  console.log(`Persona: ${persona.identity.label}  (${persona.identity.id})`);
  console.log(`  description:  ${persona.identity.description}`);
  console.log(`  declared:     ${persona.identity.declared.country}/${persona.identity.declared.region ?? "?"} tz=${persona.identity.declared.timezone}`);
  console.log(`  client:       ${persona.identity.client.os} / ${persona.identity.client.browser} v${persona.identity.client.browserVersion}`);
  console.log(`  interests:    ${persona.identity.interests.join(", ")}`);
  console.log(`  tags:         ${persona.identity.tags.join(", ")}`);
  console.log(`  fingerprint:  ua=${persona.fingerprint.userAgent.slice(0, 60)}…`);
  console.log(`  proxy:        ${persona.proxy.provider} session=${persona.proxy.sessionId}${persona.proxy.burned ? " [BURNED]" : ""}`);
  console.log(`  probes:       ${persona.operational.totalProbes}, conv: ${persona.operational.totalConversations}`);
  console.log(`  ads seen:     ${persona.history.adsSeen.total}`);
  console.log(`  healthScore:  ${persona.operational.healthScore}`);
  console.log(`  lastUsed:     ${persona.operational.lastUsed}`);
  console.log(`  on disk:      ${personaDir(id)}`);
  console.log("");
}

async function cmdHealth(): Promise<void> {
  const idx = await readEnrichedIndex();
  const ids = Object.keys(idx.personas).sort();
  if (ids.length === 0) {
    console.log("No personas to check.");
    return;
  }
  console.log("");
  for (const id of ids) {
    const p = idx.personas[id]!;
    const score = p.healthScore;
    const badge = score >= 80 ? "✅" : score >= 50 ? "⚠️ " : "❌";
    console.log(
      `${badge}  ${pad(p.id, 28)} ${pad(p.label, 28)} score=${pad(String(score), 4)} health=${pad(p.health, 16)} plan=${p.plan} probes=${p.totalProbes}`,
    );
  }
  console.log("");
  console.log("(use --validate <id> to refresh health)");
  console.log("");
}

async function cmdCreate(pm: PersonaManager, seedId: string): Promise<void> {
  const seed = getSeed(seedId);
  if (!seed) {
    console.error(`Unknown seed: ${seedId}\nAvailable: ${PERSONA_SEEDS.map((s) => s.id).join(", ")}`);
    process.exit(1);
  }
  const overrideId = arg("--id");
  const tagsArg = arg("--tags");
  const extraTags = tagsArg ? tagsArg.split(",").map((t) => t.trim()).filter(Boolean) : [];
  const dryRun = flag("--dry-run");

  const id = asPersonaId(overrideId ?? seed.id);
  if (await identityExists(id)) {
    console.error(`Persona already exists: ${id}\nUse --id to mint a different one, or --archive first.`);
    process.exit(1);
  }

  const { persona, credentials, auth } = makePersonaFromSeed(seedId, {
    id: overrideId,
    tags: extraTags,
    createdBy: process.env.USER ?? "cli",
  });

  if (dryRun) {
    console.log(`\n(dry-run) Would create persona "${id}" from seed "${seedId}"`);
    console.log(`  label:     ${persona.identity.label}`);
    console.log(`  proxy:     ${persona.proxy.provider}`);
    console.log(`  tags:      ${persona.identity.tags.join(", ")}`);
    console.log(`  path:      ${personaDir(id)}`);
    console.log("");
    return;
  }

  await pm.persistPersona(persona);
  await pm.persistCredentials(id, credentials);
  await pm.persistAuth(id, auth);
  await pm.audit(id, "created", `from seed ${seedId}`);

  console.log(`\nCreated persona "${persona.identity.label}" (${id})`);
  console.log(`  on disk:  ${personaDir(id)}`);
  console.log(`  next:     pnpm personas --status ${id}`);
  console.log(`            pnpm personas --validate ${id}  (requires browser fallback in Phase 3)`);
  console.log("");
}

async function cmdDump(pm: PersonaManager, id: PersonaId): Promise<void> {
  const persona = await pm.load(id);
  const redacted = redactPersona(persona);
  console.log(JSON.stringify(redacted, null, 2));
}

async function cmdAudit(pm: PersonaManager, id: PersonaId, limitArg?: string): Promise<void> {
  const limit = limitArg ? parseInt(limitArg, 10) : 25;
  const entries = await pm.readRecentAudit(id, limit);
  if (entries.length === 0) {
    console.log("(no audit entries)");
    return;
  }
  for (const e of entries) {
    console.log(
      `${e.ts}  ${pad(e.action, 28)}${e.reason ? "  " + e.reason : ""}`,
    );
  }
}

async function cmdArchive(pm: PersonaManager, id: PersonaId): Promise<void> {
  const { updateIndex } = await import("./storage.js");
  await updateIndex((idx) => {
    delete idx.personas[id];
    return idx;
  });
  await pm.audit(id, "archived");
  console.log(`Archived ${id} (files preserved at ${personaDir(id)})`);
}

async function cmdCreateAccount(pm: PersonaManager, seedId: string): Promise<void> {
  console.warn(
    "\n⚠️  This command launches Chromium with automation plugins.",
    "It will trip EDR products. Use only on a machine where this is OK.\n",
  );
  if (!getSeed(seedId)) {
    console.error(`Unknown seed: ${seedId}\nAvailable: ${PERSONA_SEEDS.map((s) => s.id).join(", ")}`);
    process.exit(1);
  }
  const overrideId = arg("--id");
  const tagsArg = arg("--tags");
  const extraTags = tagsArg ? tagsArg.split(",").map((t) => t.trim()).filter(Boolean) : [];
  const result = await createWithBrowser(seedId, pm, { id: overrideId, tags: extraTags });
  console.log(`\n✅ Created account for "${result.label}" (${result.id})`);
  console.log(`  on disk:  ${personaDir(result.id)}`);
  console.log(`  next:     pnpm personas --validate ${result.id}`);
  console.log("");
}

async function cmdRefreshCf(pm: PersonaManager, id: PersonaId): Promise<void> {
  console.warn("\n⚠️  This command launches Chromium with automation plugins.\n");
  await pm.recoverWithBrowser(id, "cf_blocked");
  console.log(`✅ Refreshed cf_clearance for ${id}`);
}

async function cmdReauth(pm: PersonaManager, id: PersonaId): Promise<void> {
  console.warn("\n⚠️  This command launches Chromium with automation plugins.\n");
  await pm.recoverWithBrowser(id, "session_expired");
  console.log(`✅ Re-authed ${id}`);
}

async function cmdRotateProxy(pm: PersonaManager, id: PersonaId): Promise<void> {
  if (!pm.proxyPool) {
    console.error("No proxy pool configured. Set OXYLABS_PROXY_USERNAME/PASSWORD in .env");
    process.exit(1);
  }
  await pm.rotateProxy(id);
  const assignment = pm.proxyPool.get(id);
  console.log(`✅ Rotated proxy for ${id}`);
  console.log(`   new session: ${assignment?.sessionId}`);
  console.log(`   description: ${assignment?.description}`);
}

async function cmdProxyInfo(pm: PersonaManager, idArg: string | undefined): Promise<void> {
  if (!pm.proxyPool) {
    console.error("No proxy pool configured. Set OXYLABS_PROXY_USERNAME/PASSWORD in .env");
    process.exit(1);
  }
  console.log("\nProxy pool configuration:");
  console.log(`  type:    ${pm.proxyPool.opts.type ?? "isp"}`);
  console.log(`  country: ${pm.proxyPool.opts.country ?? "US"}`);
  if (pm.proxyPool.opts.state) console.log(`  state:   ${pm.proxyPool.opts.state}`);
  if (pm.proxyPool.opts.city) console.log(`  city:    ${pm.proxyPool.opts.city}`);
  console.log(`  ttl:     ${pm.proxyPool.opts.sessionTtlMin ?? 30} min`);
  console.log(`  user:    ${pm.proxyPool.opts.oxylabsUsername ? "set" : "MISSING"}`);
  console.log(`  pass:    ${pm.proxyPool.opts.oxylabsPassword ? "set" : "MISSING"}`);
  console.log("");

  if (!idArg) return;
  const id = asPersonaId(idArg);
  const a = pm.proxyPool.get(id);
  if (!a) {
    console.log(`(no assignment for ${id} — will be created on first use)`);
    return;
  }
  console.log(`Assignment for ${id}:`);
  console.log(`  sessionId:    ${a.sessionId}`);
  console.log(`  description:  ${a.description}`);
  console.log(`  burned:       ${a.burned}${a.burnedReason ? ` (${a.burnedReason})` : ""}`);
  if (a.lastEgress) {
    console.log(`  last egress:  ${a.lastEgress.ip} (${a.lastEgress.country}${a.lastEgress.region ? "/" + a.lastEgress.region : ""})`);
  }
  console.log(`  proxy URL:    ${pm.proxyUrlFor(id)?.replace(/:[^:@]+@/, ":***@") ?? "(none)"}`);
  console.log("");
}

async function cmdProxyTest(proxyPool: import("./proxy/pool.js").ProxyPool): Promise<void> {
  if (!proxyPool.opts.oxylabsUsername) {
    console.error("No proxy creds. Set OXYLABS_PROXY_USERNAME / OXYLABS_PROXY_PASSWORD in .env");
    process.exit(1);
  }
  // Build a throwaway persona id for the test assignment.
  const testId = asPersonaId("__proxy_test__");
  const a = proxyPool.assign(testId);
  const url = proxyPool.proxyUrlFor(a);
  if (!url) {
    console.error("Proxy pool is disabled (PERSONA_PROXY_DISABLED=true)");
    process.exit(1);
  }
  // Use Node's undici via fetch with a Proxy agent. Easiest: use the
  // dynamic import of node's URL + a HTTPS-over-HTTP-CONNECT tunnel.
  // For simplicity, use fetch with dispatcher (Node 22+ supports this).
  // Fall back to: print the URL and ask the user to verify externally.
  console.log(`Testing proxy: ${url.replace(/:[^:@]+@/, ":***@")}`);
  console.log("Hitting https://ip.oxylabs.io/location through the proxy...");
  // Tunnel via http(s) request through the proxy
  try {
    const { request } = await import("node:https");
    const { response, body } = await tunneledGet(url, "https://ip.oxylabs.io/location");
    if (response.statusCode === 200) {
      const data = JSON.parse(body.toString("utf8"));
      console.log("✅ Proxy is working. Observed egress:");
      console.log(`   ip:        ${data.ip ?? "?"}`);
      console.log(`   country:   ${data.country ?? "?"}`);
      console.log(`   region:    ${data.region ?? "?"}`);
      console.log(`   city:      ${data.city ?? "?"}`);
      console.log(`   timezone:  ${data.timezone ?? "?"}`);
    } else {
      console.log(`❌ Proxy responded ${response.statusCode}`);
      console.log(body.toString("utf8").slice(0, 200));
    }
  } catch (e) {
    console.log(`❌ Proxy test failed: ${e instanceof Error ? e.message : e}`);
    console.log(`   URL was: ${url.replace(/:[^:@]+@/, ":***@")}`);
    process.exit(1);
  }
}

/** Tunnel a GET through an HTTP proxy (HTTP-CONNECT for HTTPS, absolute-URI for HTTP). */
async function tunneledGet(
  proxyUrl: string,
  target: string,
): Promise<{ response: { statusCode: number }; body: Buffer }> {
  const { request } = await import("node:http");
  const proxy = new URL(proxyUrl);
  const tu = new URL(target);
  // For HTTPS targets, use CONNECT; for HTTP, send absolute URI.
  if (tu.protocol === "https:") {
    return new Promise((resolve, reject) => {
      const req = request({
        host: proxy.hostname,
        port: proxy.port,
        method: "CONNECT",
        path: `${tu.hostname}:${tu.port || 443}`,
        headers: {
          "Proxy-Authorization":
            "Basic " +
            Buffer.from(
              decodeURIComponent(proxy.username) + ":" + decodeURIComponent(proxy.password),
            ).toString("base64"),
        },
      });
      req.on("connect", (res, socket) => {
        if (res.statusCode !== 200) {
          reject(new Error(`CONNECT ${res.statusCode}`));
          return;
        }
        // Tunneled TLS over the socket
        const tls = require("node:tls") as typeof import("node:tls");
        const ts = tls.connect({
          host: tu.hostname,
          port: tu.port ? parseInt(tu.port, 10) : 443,
          socket,
        });
        let buf = Buffer.alloc(0);
        let headerEnd = -1;
        const headers =
          `GET ${tu.pathname}${tu.search} HTTP/1.1\r\n` +
          `Host: ${tu.host}\r\n` +
          `User-Agent: persona-proxy-test/1.0\r\n` +
          `Connection: close\r\n\r\n`;
        ts.write(headers);
        ts.on("data", (d) => {
          buf = Buffer.concat([buf, d]);
          if (headerEnd < 0) {
            const i = buf.indexOf("\r\n\r\n");
            if (i >= 0) {
              headerEnd = i + 4;
              const statusLine = buf.slice(0, buf.indexOf("\r\n")).toString("utf8");
              const m = statusLine.match(/^HTTP\/\d\.\d (\d+)/);
              const statusCode = m ? parseInt(m[1]!, 10) : 0;
              const body = buf.slice(headerEnd);
              resolve({ response: { statusCode }, body });
            }
          }
        });
        ts.on("end", () => {
          if (headerEnd < 0) {
            // No headers parsed yet; treat as empty body
            resolve({ response: { statusCode: 0 }, body: buf });
          }
        });
        ts.on("error", reject);
      });
      req.on("error", reject);
      req.end();
    });
  }
  throw new Error("HTTP target not yet supported in --proxy-test");
}

async function cmdValidate(pm: PersonaManager, id: PersonaId): Promise<void> {
  // Use the no-fallback variant: --validate is for *introspection*;
  // we don't want a stray browser launch just because a session is dead.
  let auth: import("./types.js").AuthState;
  try {
    auth = await pm.validateNoFallback(id);
  } catch (e: any) {
    const code = e?.code ?? "unknown";
    const msg = e instanceof Error ? e.message : String(e);
    console.log(`❌ ${id}: ${code} — ${msg}`);
    if (code === "cf_blocked") {
      console.log(`   run: pnpm personas --refresh-cf ${id}   (requires browser)`);
    } else if (code === "session_expired") {
      console.log(`   run: pnpm personas --reauth ${id}   (requires browser)`);
    }
    return;
  }
  // Probe /me if we have an access token.
  let plan = auth.plan;
  let email = "?";
  if (auth.accessToken) {
    const client = new ChatGPTClient(
      await pm.load(id),
      auth,
    );
    try {
      const me = await client.getMe();
      plan = me.plan;
      email = me.email;
    } catch {
      /* /me may fail; keep the cached plan */
    }
  }
  console.log(
    `✅ ${id}: health=${auth.health} accountState=${auth.accountState} plan=${plan} email=${email}`,
  );
}

async function cmdProbe(pm: PersonaManager, id: PersonaId, prompt: string): Promise<void> {
  const client = (await pm.getClient(id)) as ChatGPTClient;
  const result = await client.probe(prompt);
  console.log("");
  console.log(`prompt:    ${prompt.slice(0, 80)}${prompt.length > 80 ? "…" : ""}`);
  console.log(`model:     ${result.model}`);
  console.log(`elapsed:   ${result.elapsedMs}ms`);
  console.log(`conv id:   ${result.conversationId}`);
  console.log(`msg id:    ${result.messageId}`);
  console.log(`ads found: ${result.ads.length}`);
  for (const ad of result.ads) {
    console.log(`  🔴 ${ad.advertiser || "?"}: "${ad.title}"`);
  }
  console.log(`text:      ${result.text.slice(0, 200)}${result.text.length > 200 ? "…" : ""}`);
  console.log("");
}

async function cmdConverse(pm: PersonaManager, id: PersonaId, file?: string): Promise<void> {
  if (!file) {
    console.error("Usage: pnpm personas --converse <id> <file>");
    process.exit(1);
  }
  const prompts = (await readFile(file, "utf8"))
    .split("\n")
    .map((l) => l.trim())
    .filter((l) => l.length > 0);
  const client = (await pm.getClient(id)) as ChatGPTClient;
  let conversationId: import("./types.js").ConversationId | undefined = undefined;
  for (let i = 0; i < prompts.length; i++) {
    process.stdout.write(`[${i + 1}/${prompts.length}] `);
    try {
      const result = await client.probe(prompts[i]!, { conversationId });
      if (result.conversationId) conversationId = result.conversationId;
      console.log(
        `ads=${result.ads.length} t=${result.elapsedMs}ms${result.ads.length > 0 ? ` 🔴 ${result.ads[0]?.advertiser}: "${result.ads[0]?.title}"` : ""}`,
      );
    } catch (e) {
      console.log(`✗ ${e instanceof Error ? e.message.slice(0, 80) : e}`);
    }
  }
}

async function cmdBatch(pm: PersonaManager, id: PersonaId, file?: string): Promise<void> {
  if (!file) {
    console.error("Usage: pnpm personas --batch <id> <file>");
    process.exit(1);
  }
  const prompts = (await readFile(file, "utf8"))
    .split("\n")
    .map((l) => l.trim())
    .filter((l) => l.length > 0);
  const client = (await pm.getClient(id)) as ChatGPTClient;
  let totalAds = 0;
  for (let i = 0; i < prompts.length; i++) {
    process.stdout.write(`[${i + 1}/${prompts.length}] `);
    try {
      const result = await client.probe(prompts[i]!);
      totalAds += result.ads.length;
      console.log(
        `ads=${result.ads.length} t=${result.elapsedMs}ms${result.ads.length > 0 ? ` 🔴 ${result.ads[0]?.advertiser}: "${result.ads[0]?.title}"` : ""}`,
      );
    } catch (e) {
      console.log(`✗ ${e instanceof Error ? e.message.slice(0, 80) : e}`);
    }
  }
  console.log(`\nDONE. ${totalAds} ads across ${prompts.length} prompts.`);
}

async function cmdImportSession(
  pm: PersonaManager,
  id: PersonaId,
  token: string,
  cf?: string,
): Promise<void> {
  const { randomUUID } = await import("node:crypto");
  const existing = await pm.loadAuth(id).catch(() => null);
  const deviceId = existing?.deviceId ?? randomUUID();
  const auth: AuthState = {
    sessionToken: token,
    accessToken: null,
    accessTokenExp: null,
    cfClearance: cf ?? null,
    cfClearanceExp: cf ? Math.floor(Date.now() / 1000) + 30 * 24 * 60 * 60 : null,
    deviceId,
    puid: null,
    lastValidatedAt: new Date().toISOString(),
    health: "unknown",
    sessionStartedAt: new Date().toISOString(),
    accountState: "UNKNOWN",
    plan: "unknown",
  };
  await pm.persistAuth(id, auth);
  await pm.audit(id, "session_obtained", "manual import");
  console.log(`✅ Session imported for ${id}. Run pnpm personas --validate ${id} to verify.`);
}

function pad(s: string, n: number): string {
  return s.length >= n ? s.slice(0, n) : s + " ".repeat(n - s.length);
}

/** Get the Nth positional argument (1-indexed), after all flags. */
function positional(n: number): string | undefined {
  const args = process.argv.slice(2);
  let i = 0;
  let seen = 0;
  while (i < args.length) {
    const a = args[i]!;
    if (a.startsWith("--")) {
      // skip this flag + its value (if next arg doesn't start with --)
      if (i + 1 < args.length && !args[i + 1]!.startsWith("--")) i += 2;
      else i += 1;
      continue;
    }
    seen++;
    if (seen === n) return a;
    i++;
  }
  return undefined;
}

function redactPersona(p: import("./types.js").Persona): unknown {
  return {
    ...p,
    // redact anything sensitive
    fingerprint: { ...p.fingerprint, userAgent: p.fingerprint.userAgent.slice(0, 30) + "…" },
  };
}

main().catch((e) => {
  console.error("personas error:", e instanceof Error ? e.message : e);
  process.exit(1);
});
