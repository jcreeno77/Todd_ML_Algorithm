# Notifications

## Status: Twilio removed; logging is the default channel

Twilio/WhatsApp alerting has been **fully removed** from the codebase. All alerts
(trade buy/sell signals, Schwab refresh-token expiry warnings, collector errors)
now flow through a single channel-agnostic notifier:

```
ML_tradingAlgo/data/notify.py  ->  notify(message, level="info") -> "webhook" | "log"
```

With no configuration, every alert is **logged** — zero external dependency,
nothing breaks if alerting is unconfigured. `notify` is the single seam, so the
delivery channel can change without touching any call site.

### Call sites routed through `notify`
- `live_data_gather_unified.py` — buy signal, SOLD-2 (partial exit), SOLD-3 (full exit)
- `data/collector.py` — Schwab refresh-token freshness (via `token_health.check_token_freshness(notify=notify)`)
- `data/token_health.py` — pluggable `notify` callback (defaults to `logger.warning`)

## Planned: switch to a Discord webhook (production channel)

The intended production channel is a **Discord webhook**. The notifier already
supports it — it is dormant until configured:

1. In Discord: *Server Settings → Integrations → Webhooks → New Webhook*, copy the
   webhook URL.
2. Set it in `ML_tradingAlgo/.env` (gitignored):
   ```
   ALERT_WEBHOOK_URL=https://discord.com/api/webhooks/XXXX/YYYY
   ```
3. Done. `notify()` will additionally POST `{"content": <message>}` to the webhook
   on every alert. Delivery is best-effort: a webhook failure degrades to a logged
   warning and never crashes the trading loop.

### Why Discord (vs Slack / SMS)
- Free, no per-message cost (unlike Twilio SMS/WhatsApp), no phone-number provisioning.
- Simple unauthenticated webhook POST — no OAuth, no SDK dependency.
- Mobile push notifications out of the box.

### Slack compatibility
Slack incoming webhooks use `{"text": <message>}` instead of Discord's
`{"content": <message>}`. To support Slack, branch the payload in
`notify._http_post` on URL host (or a `ALERT_WEBHOOK_FLAVOR` env var). Not built
yet — Discord is the planned target.

## Future enhancements (not built)
- Severity routing (e.g. errors → @here, info → plain message).
- Rate limiting / batching to avoid Discord's per-webhook rate limits during busy sessions.
- A second channel for the live dashboard's stale-state warnings.
