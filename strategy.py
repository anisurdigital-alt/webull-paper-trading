#!/usr/bin/env python3
"""
Small-Cap Momentum Scanner + Paper Trade Execution
====================================================

Fully mechanical version of the original Claude-run strategy, rewritten to run
unattended inside a GitHub Actions workflow instead of an interactive Claude
session. This removes the dependency on a laptop being on / connected, at the
cost of turning the entry-pattern checks (Step 2) from LLM judgment into hard
Python rules on OHLCV bars.

SAFETY DESIGN NOTES (read before changing anything):
  * The Webull API host is READ FROM AN ENV VAR but DEFAULTS to the sandbox
    host, and the script hard-refuses to run unless the resolved host
    contains "sandbox" -- see `assert_sandbox()`. This makes it structurally
    hard to point this at a live-money endpoint by accident.
  * DRY_RUN=true (the default unless explicitly set to "false") runs the full
    pipeline -- scan, pattern check, sizing, all safety checks -- but calls
    `preview_order` instead of `place_order`, and logs everything to Notion
    with Action="DryRun". Flip DRY_RUN to "false" only after you've watched a
    few dry runs and are comfortable with what it's deciding.
  * This script was written and unit-tested for import/syntax correctness,
    but COULD NOT be run end-to-end against api.sandbox.webull.com from the
    environment that built it (outbound network to that host was blocked
    there). Treat the first several scheduled runs as the real test --
    watch the Actions logs and the Notion log closely.

Trading window: restricted to 9:30 AM - 11:00 AM ET (regular market hours
only). The original spec wanted 4:00 AM premarket too, but Webull's
"support_trading_session" values for premarket/extended-hours orders could
not be confirmed from documentation available at build time, so premarket
was deliberately left out rather than guessed. See README.md.
"""

import json
import logging
import os
import sys
import uuid
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import requests

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    stream=sys.stdout,
)
log = logging.getLogger("momentum-scanner")

ET = ZoneInfo("America/New_York")

# ---------------------------------------------------------------------------
# Config (from environment / GitHub Actions secrets)
# ---------------------------------------------------------------------------

APP_KEY = os.environ["WEBULL_APP_KEY"]
APP_SECRET = os.environ["WEBULL_APP_SECRET"]
REGION = os.environ.get("WEBULL_REGION", "us")
API_HOST = os.environ.get("WEBULL_API_HOST", "api.sandbox.webull.com")

NOTION_TOKEN = os.environ["NOTION_TOKEN"]
NOTION_DATA_SOURCE_ID = os.environ["NOTION_DATA_SOURCE_ID"]

DRY_RUN = os.environ.get("DRY_RUN", "true").lower() != "false"

MAX_DAILY_LOSS = -100.0
MAX_CONSECUTIVE_LOSSES = 3
MAX_OPEN_POSITIONS = 3
RISK_PCT = 0.02
REWARD_RISK_RATIO = 2.0
MIN_PRICE, MAX_PRICE = 2.0, 20.0
MIN_CHANGE_PCT = 10.0
MIN_RELATIVE_VOLUME = 5.0
MAX_SPREAD_PCT = 5.0


def assert_sandbox():
    if "sandbox" not in API_HOST.lower():
        raise SystemExit(
            f"REFUSING TO RUN: WEBULL_API_HOST='{API_HOST}' does not look like a "
            f"sandbox/paper endpoint. This script will only run against a host "
            f"containing 'sandbox'. Aborting without making any Webull calls."
        )


def now_et():
    return datetime.now(ET)


def in_trading_window(t):
    if t.weekday() >= 5:  # Sat/Sun
        return False, "weekend"
    start = t.replace(hour=9, minute=30, second=0, microsecond=0)
    end = t.replace(hour=11, minute=0, second=0, microsecond=0)
    if t < start or t > end:
        return False, f"outside 9:30-11:00 ET window (now {t.strftime('%H:%M')} ET)"
    return True, None


# ---------------------------------------------------------------------------
# Notion logging (direct HTTPS to api.notion.com -- no MCP needed)
# ---------------------------------------------------------------------------

NOTION_API = "https://api.notion.com/v1"
NOTION_HEADERS = {
    "Authorization": f"Bearer {NOTION_TOKEN}",
    "Notion-Version": "2025-09-03",
    "Content-Type": "application/json",
}


def notion_log_row(*, ticker, pattern, entry=None, stop=None, target=None,
                    shares=None, action, reason, outcome="", time_et=None):
    """Insert one audit row. Never raises -- logging failures must not crash a
    trading run; they just get printed to the Actions log instead."""
    t = time_et or now_et()
    name = f"{ticker} {t.strftime('%H:%M')} ET"
    props = {
        "Name": {"title": [{"text": {"content": name}}]},
        "Date": {"date": {"start": t.strftime("%Y-%m-%d")}},
        "Time ET": {"rich_text": [{"text": {"content": t.strftime("%H:%M:%S")}}]},
        "Ticker": {"rich_text": [{"text": {"content": ticker}}]},
        "Action": {"select": {"name": action}},
        "Reason": {"rich_text": [{"text": {"content": (reason or "")[:1900]}}]},
        "Outcome": {"rich_text": [{"text": {"content": (outcome or "")[:1900]}}]},
    }
    if pattern:
        props["Pattern"] = {"select": {"name": pattern}}
    if entry is not None:
        props["Entry"] = {"number": round(float(entry), 4)}
    if stop is not None:
        props["Stop"] = {"number": round(float(stop), 4)}
    if target is not None:
        props["Target"] = {"number": round(float(target), 4)}
    if shares is not None:
        props["Shares"] = {"number": int(shares)}

    body = {
        "parent": {"type": "data_source_id", "data_source_id": NOTION_DATA_SOURCE_ID},
        "properties": props,
    }
    try:
        resp = requests.post(f"{NOTION_API}/pages", headers=NOTION_HEADERS, json=body, timeout=15)
        if resp.status_code >= 300:
            log.error("Notion log insert failed (%s): %s", resp.status_code, resp.text[:500])
        else:
            log.info("Logged to Notion: %s | %s | %s", ticker, action, reason)
    except Exception as e:
        log.error("Notion log insert raised: %r", e)


def notion_query_today():
    """Return today's rows (best-effort; used only for a visible audit
    cross-check, not as the source of truth -- Webull's own order history is
    the source of truth for P&L / streaks in this script)."""
    today = now_et().strftime("%Y-%m-%d")
    body = {
        "filter": {"property": "Date", "date": {"equals": today}},
        "page_size": 100,
    }
    try:
        resp = requests.post(
            f"{NOTION_API}/data_sources/{NOTION_DATA_SOURCE_ID}/query",
            headers=NOTION_HEADERS, json=body, timeout=15,
        )
        if resp.status_code >= 300:
            log.warning("Notion query failed (%s): %s", resp.status_code, resp.text[:300])
            return []
        return resp.json().get("results", [])
    except Exception as e:
        log.warning("Notion query raised: %r", e)
        return []


# ---------------------------------------------------------------------------
# Webull client setup
# ---------------------------------------------------------------------------

def build_clients():
    from webull.core.client import ApiClient
    from webull.trade.trade_client import TradeClient
    from webull.data.data_client import DataClient

    api_client = ApiClient(APP_KEY, APP_SECRET, REGION)
    api_client.add_endpoint(REGION, API_HOST)
    trade_client = TradeClient(api_client)
    data_client = DataClient(api_client)
    return trade_client, data_client


def get_account_id(trade_client):
    res = trade_client.account_v2.get_account_list()
    log.info("account_list raw response: %s", res.text[:1000])
    if res.status_code != 200:
        raise RuntimeError(f"get_account_list failed: {res.status_code} {res.text[:300]}")
    data = res.json()
    accounts = data if isinstance(data, list) else data.get("data") or data.get("accounts") or []
    if not accounts:
        raise RuntimeError(f"No accounts returned: {data}")
    acct = accounts[0]
    account_id = acct.get("account_id") or acct.get("accountId") or acct.get("id")
    if not account_id:
        raise RuntimeError(f"Could not find account_id field in: {acct}")
    log.info("Using account_id=%s", account_id)
    return account_id


# ---------------------------------------------------------------------------
# Step 0 - account state / halt conditions
# ---------------------------------------------------------------------------

def _num(x, default=0.0):
    try:
        return float(x)
    except (TypeError, ValueError):
        return default


def get_account_state(trade_client, account_id):
    balance_res = trade_client.account_v2.get_account_balance(account_id)
    log.info("account_balance raw response: %s", balance_res.text[:1500])
    balance_json = balance_res.json() if balance_res.status_code == 200 else {}

    positions_res = trade_client.account_v2.get_account_position(account_id)
    log.info("positions raw response: %s", positions_res.text[:1500])
    positions_json = positions_res.json() if positions_res.status_code == 200 else {}
    positions = positions_json if isinstance(positions_json, list) else positions_json.get("data") or positions_json.get("positions") or []

    open_orders_res = trade_client.order_v3.get_order_open(account_id, page_size=50)
    log.info("open_orders raw response: %s", open_orders_res.text[:1500])
    open_orders_json = open_orders_res.json() if open_orders_res.status_code == 200 else {}
    open_orders = open_orders_json if isinstance(open_orders_json, list) else open_orders_json.get("data") or open_orders_json.get("orders") or []

    today = now_et().strftime("%Y-%m-%d")
    history_res = trade_client.order_v3.get_order_history(account_id, page_size=100, start_date=today, end_date=today)
    log.info("order_history raw response: %s", history_res.text[:2000])
    history_json = history_res.json() if history_res.status_code == 200 else {}
    history = history_json if isinstance(history_json, list) else history_json.get("data") or history_json.get("orders") or []

    # Best-effort extraction of cash/net-liquidation value; field names not
    # confirmed from live docs, so try several common candidates and log
    # whatever we actually got so a human can sanity-check it.
    balance_val = None
    for container in ([balance_json] if isinstance(balance_json, dict) else balance_json):
        if not isinstance(container, dict):
            continue
        for key in ("net_liquidation", "netLiquidation", "total_asset", "totalAsset",
                    "cash_balance", "cashBalance", "buying_power", "buyingPower"):
            if key in container:
                balance_val = _num(container[key])
                break
        if balance_val is not None:
            break

    # Realized P&L today and consecutive-loss streak, derived from filled
    # orders in today's history. Field names are best-effort (see above).
    filled = [o for o in history if str(o.get("status", "")).upper() in ("FILLED", "DONE", "EXECUTED")]
    realized_pnl = 0.0
    streak = 0
    max_streak = 0
    for o in sorted(filled, key=lambda o: o.get("filled_time") or o.get("updated_time") or ""):
        pnl = o.get("realized_pnl") or o.get("realizedPnl")
        if pnl is None:
            continue
        pnl = _num(pnl)
        realized_pnl += pnl
        if pnl < 0:
            streak += 1
            max_streak = max(max_streak, streak)
        else:
            streak = 0

    return {
        "balance": balance_val,
        "positions": positions,
        "open_orders": open_orders,
        "history": history,
        "realized_pnl_today": realized_pnl,
        "consecutive_losses": streak,
        "raw_balance_json": balance_json,
    }


def check_halt(state):
    reasons = []
    if state["realized_pnl_today"] <= MAX_DAILY_LOSS:
        reasons.append(f"realized P&L today {state['realized_pnl_today']:.2f} <= {MAX_DAILY_LOSS}")
    if state["consecutive_losses"] >= MAX_CONSECUTIVE_LOSSES:
        reasons.append(f"{state['consecutive_losses']} consecutive losing trades today")
    ok, window_reason = in_trading_window(now_et())
    if not ok:
        reasons.append(window_reason)
    return reasons


# ---------------------------------------------------------------------------
# Step 1 - scan for candidates
# ---------------------------------------------------------------------------

def scan_candidates(data_client):
    gainers_res = data_client.screener.get_gainers_losers(
        rank_type="DAY_1", category="US_STOCK", sort_by="CHANGE_RATIO",
        direction="DESC", page_size=50,
    )
    log.info("gainers screener raw response: %s", gainers_res.text[:2000])
    gainers_json = gainers_res.json() if gainers_res.status_code == 200 else {}
    gainers = gainers_json if isinstance(gainers_json, list) else gainers_json.get("data") or gainers_json.get("list") or []

    active_res = data_client.screener.get_most_active(
        category="US_STOCK", rank_type="RELATIVE_VOLUME_10D",
        sort_by="RELATIVE_VOLUME_10D", direction="DESC", page_size=50,
    )
    log.info("most-active(rel-vol) screener raw response: %s", active_res.text[:2000])
    active_json = active_res.json() if active_res.status_code == 200 else {}
    active = active_json if isinstance(active_json, list) else active_json.get("data") or active_json.get("list") or []
    rel_vol_by_symbol = {}
    for row in active:
        sym = row.get("symbol")
        rv = row.get("relative_volume") or row.get("RELATIVE_VOLUME_10D") or row.get("rel_volume")
        if sym and rv is not None:
            rel_vol_by_symbol[sym] = _num(rv)

    candidates = []
    for row in gainers:
        symbol = row.get("symbol")
        change_pct = _num(row.get("change_ratio") or row.get("changeRatio") or row.get("change_pct"))
        price = _num(row.get("price") or row.get("close"))
        if not symbol:
            continue
        log.info("Candidate raw data: %s change=%.2f%% price=%.2f", symbol, change_pct, price)

        if price <= 0 or change_pct == 0:
            notion_log_row(ticker=symbol, action="Anomaly", pattern=None,
                            reason=f"implausible data: price={price} change={change_pct}")
            continue
        if change_pct < MIN_CHANGE_PCT:
            continue
        if not (MIN_PRICE <= price <= MAX_PRICE):
            continue

        rel_vol = rel_vol_by_symbol.get(symbol)
        if rel_vol is None:
            notion_log_row(ticker=symbol, action="Skipped", pattern=None,
                            reason="relative volume unavailable for this ticker")
            continue
        if rel_vol < MIN_RELATIVE_VOLUME:
            continue

        candidates.append({"symbol": symbol, "price": price, "change_pct": change_pct, "rel_vol": rel_vol})

    log.info("Float/shares-outstanding filter: not exposed by Webull OpenAPI (confirmed at build "
             "time -- no fundamentals endpoint returns float or shares outstanding). Filter skipped "
             "for all tickers per the strategy's own fallback rule.")

    return candidates


def check_spread_and_type(data_client, symbol):
    """Returns (ok, reason, quote_dict). Filters OTC/warrant/wide-spread."""
    res = data_client.market_data.get_quotes(symbol, category="US_STOCK", depth="L1")
    log.info("quote raw response for %s: %s", symbol, res.text[:800])
    if res.status_code != 200:
        return False, f"quote fetch failed ({res.status_code})", None
    q = res.json()
    q = q[0] if isinstance(q, list) and q else q
    if not isinstance(q, dict):
        return False, "unexpected quote shape", None

    bid = _num(q.get("bid_price") or q.get("bidPrice"))
    ask = _num(q.get("ask_price") or q.get("askPrice"))
    if bid <= 0 or ask <= 0:
        return False, "missing/implausible bid or ask", q
    spread_pct = (ask - bid) / ask * 100
    if spread_pct > MAX_SPREAD_PCT:
        return False, f"spread {spread_pct:.2f}% > {MAX_SPREAD_PCT}%", q

    exch = str(q.get("exchange") or q.get("exchangeCode") or "").upper()
    if "OTC" in exch:
        return False, "OTC-listed", q

    return True, None, q


# ---------------------------------------------------------------------------
# Step 2 - entry pattern detection (hard-coded rules on 5-min bars)
# ---------------------------------------------------------------------------

def get_bars(data_client, symbol, count=60):
    res = data_client.market_data.get_history_bar(symbol, category="US_STOCK", timespan="M5", count=str(count))
    log.info("bars raw response for %s: %s", symbol, res.text[:1200])
    if res.status_code != 200:
        return []
    data = res.json()
    bars = data if isinstance(data, list) else data.get("data") or data.get("bars") or []
    out = []
    for b in bars:
        try:
            out.append({
                "o": _num(b.get("open") or b.get("o")),
                "h": _num(b.get("high") or b.get("h")),
                "l": _num(b.get("low") or b.get("l")),
                "c": _num(b.get("close") or b.get("c")),
                "v": _num(b.get("volume") or b.get("v")),
                "t": b.get("timestamp") or b.get("t"),
            })
        except Exception:
            continue
    return out


def detect_micro_pullback(bars):
    """Shallow pullback (<50% retrace of last up-move) followed by a break of
    the pullback candle's high. Looks at the most recent ~15 bars."""
    if len(bars) < 6:
        return None
    window = bars[-15:]
    # find the strongest recent up-move: lowest low -> highest high after it
    low_idx = min(range(len(window)), key=lambda i: window[i]["l"])
    if low_idx >= len(window) - 2:
        return None
    post = window[low_idx:]
    high_idx = max(range(len(post)), key=lambda i: post[i]["h"])
    if high_idx < 1 or high_idx >= len(post) - 1:
        return None
    move_low = post[0]["l"]
    move_high = post[high_idx]["h"]
    move_size = move_high - move_low
    if move_size <= 0:
        return None
    # exclude the final bar -- that's the one being tested for the breakout,
    # not part of the pullback itself
    pullback_bars = post[high_idx + 1:-1]
    if not pullback_bars:
        return None
    pullback_low = min(b["l"] for b in pullback_bars)
    retrace = (move_high - pullback_low) / move_size
    if retrace >= 0.5:
        return None
    pullback_candle_high = max(b["h"] for b in pullback_bars)
    last = bars[-1]
    if last["c"] > pullback_candle_high:
        return {"pattern": "Micro-pullback", "trigger_price": pullback_candle_high}
    return None


def detect_bull_flag(bars):
    """Strong green thrust, then tight sideways/down consolidation, entry on
    break of the flag's high."""
    if len(bars) < 6:
        return None
    window = bars[-15:]
    for i in range(len(window) - 4, 0, -1):
        thrust = window[i]
        body = thrust["c"] - thrust["o"]
        rng = thrust["h"] - thrust["l"]
        if rng <= 0 or body <= 0:
            continue
        if body / rng < 0.6:
            continue
        avg_range = sum(b["h"] - b["l"] for b in window[max(0, i - 5):i]) / max(1, min(5, i))
        if avg_range <= 0 or rng < avg_range * 1.5:
            continue  # not a strong enough thrust vs. recent bars
        flag_bars = window[i + 1:-1]
        if len(flag_bars) < 2:
            continue
        flag_high = max(b["h"] for b in flag_bars)
        flag_low = min(b["l"] for b in flag_bars)
        if flag_high - flag_low > rng * 0.6:
            continue  # consolidation not tight enough
        if any(b["c"] > thrust["h"] for b in flag_bars):
            continue  # already broke out during the flag, this setup is stale
        last = bars[-1]
        if last["c"] > flag_high:
            return {"pattern": "Bull flag", "trigger_price": flag_high}
    return None


def detect_flat_top_breakout(bars, tolerance_pct=0.3):
    """Price has touched the same resistance level 2+ times, entry on a break
    above that level."""
    if len(bars) < 8:
        return None
    window = bars[-20:-1]
    highs = [b["h"] for b in window]
    touches = {}
    for h in highs:
        bucket = round(h / (h * tolerance_pct / 100 + 1e-9))
        matched = None
        for level in touches:
            if abs(level - h) / level * 100 <= tolerance_pct:
                matched = level
                break
        if matched is not None:
            touches[matched] += 1
        else:
            touches[h] = 1
    candidates = [(level, n) for level, n in touches.items() if n >= 2]
    if not candidates:
        return None
    level = max(candidates, key=lambda x: x[1])[0]
    last = bars[-1]
    if last["c"] > level:
        return {"pattern": "Flat top breakout", "trigger_price": level}
    return None


def detect_pattern(bars):
    for fn in (detect_micro_pullback, detect_bull_flag, detect_flat_top_breakout):
        result = fn(bars)
        if result:
            return result
    return None


# ---------------------------------------------------------------------------
# Step 4 - position sizing
# ---------------------------------------------------------------------------

def size_trade(account_balance, ask_price):
    risk_dollars = RISK_PCT * account_balance
    entry = ask_price
    stop = entry * 0.98
    risk_per_share = entry - stop
    if risk_per_share <= 0:
        return None
    shares = int(risk_dollars // risk_per_share)
    if shares <= 0:
        return None
    target_dollars = REWARD_RISK_RATIO * risk_dollars
    target = entry + (target_dollars / shares)
    return {
        "risk_dollars": risk_dollars, "entry": entry, "stop": stop,
        "risk_per_share": risk_per_share, "shares": shares,
        "target_dollars": target_dollars, "target": target,
    }


# ---------------------------------------------------------------------------
# Step 6 - place order
# ---------------------------------------------------------------------------

def place_limit_order(trade_client, account_id, symbol, shares, entry_price):
    client_order_id = uuid.uuid4().hex
    order = {
        "client_order_id": client_order_id,
        "symbol": symbol,
        "instrument_type": "EQUITY",
        "market": "US",
        "order_type": "LIMIT",
        "limit_price": str(round(entry_price, 2)),
        "quantity": str(shares),
        "support_trading_session": "CORE",
        "side": "BUY",
        "time_in_force": "DAY",
        "entrust_type": "QTY",
    }
    log.info("Order payload: %s", json.dumps(order))

    preview_res = trade_client.order_v3.preview_order(account_id, [order])
    log.info("preview_order raw response: %s", preview_res.text[:1500])
    if preview_res.status_code != 200:
        return {"placed": False, "reason": f"preview failed: {preview_res.status_code} {preview_res.text[:300]}"}

    if DRY_RUN:
        return {"placed": False, "reason": "DRY_RUN=true, order not actually placed", "client_order_id": client_order_id}

    place_res = trade_client.order_v3.place_order(account_id, [order])
    log.info("place_order raw response: %s", place_res.text[:1500])
    if place_res.status_code != 200:
        return {"placed": False, "reason": f"place failed: {place_res.status_code} {place_res.text[:300]}"}
    return {"placed": True, "client_order_id": client_order_id, "response": place_res.json()}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    assert_sandbox()
    t = now_et()
    log.info("=== Run start %s ET | DRY_RUN=%s | host=%s ===", t.isoformat(), DRY_RUN, API_HOST)

    ok, reason = in_trading_window(t)
    if not ok:
        log.info("Outside trading window (%s) -- exiting quietly, no log entry needed.", reason)
        return

    trade_client, data_client = build_clients()

    try:
        account_id = get_account_id(trade_client)
    except Exception as e:
        log.error("Could not resolve account_id: %r", e)
        notion_log_row(ticker="-", action="Halted", pattern=None, reason=f"Webull unreachable / account lookup failed: {e}")
        return

    try:
        state = get_account_state(trade_client, account_id)
    except Exception as e:
        log.error("Could not fetch account state: %r", e)
        notion_log_row(ticker="-", action="Halted", pattern=None, reason=f"Webull unreachable / account state failed: {e}")
        return

    halt_reasons = check_halt(state)
    if halt_reasons:
        reason = "; ".join(halt_reasons)
        log.warning("HALT: %s", reason)
        notion_log_row(ticker="-", action="Halted", pattern=None, reason=reason,
                       outcome=f"open positions: {len(state['positions'])}")
        return

    if len(state["positions"]) >= MAX_OPEN_POSITIONS:
        notion_log_row(ticker="-", action="Halted", pattern=None,
                       reason=f"{len(state['positions'])} open positions >= max {MAX_OPEN_POSITIONS}")
        return

    held_symbols = {p.get("symbol") for p in state["positions"] if p.get("symbol")}
    held_symbols |= {o.get("symbol") for o in state["open_orders"] if o.get("symbol")}

    candidates = scan_candidates(data_client)
    log.info("Candidates after Step 1 filters: %s", [c["symbol"] for c in candidates])

    if not candidates:
        notion_log_row(ticker="-", action="Skipped", pattern="None", reason="no qualifying candidates this scan")
        return

    took_trade = False
    for cand in candidates:
        symbol = cand["symbol"]
        if symbol in held_symbols:
            notion_log_row(ticker=symbol, action="Skipped", pattern=None,
                           reason="already holding a position or open order in this ticker today")
            continue

        ok, reason, quote = check_spread_and_type(data_client, symbol)
        if not ok:
            notion_log_row(ticker=symbol, action="Skipped", pattern=None, reason=reason)
            continue

        bars = get_bars(data_client, symbol)
        pattern = detect_pattern(bars)
        if not pattern:
            notion_log_row(ticker=symbol, action="Skipped", pattern="None",
                           reason="no micro-pullback / bull flag / flat-top-breakout pattern detected")
            continue

        if state["balance"] is None:
            notion_log_row(ticker=symbol, action="Skipped", pattern=pattern["pattern"],
                           reason="could not determine account balance from Webull response, see raw log")
            continue

        ask = _num(quote.get("ask_price") or quote.get("askPrice"))
        if ask <= 0:
            notion_log_row(ticker=symbol, action="Skipped", pattern=pattern["pattern"],
                           reason="no valid ask price")
            continue

        sizing = size_trade(state["balance"], ask)
        if not sizing:
            notion_log_row(ticker=symbol, action="Skipped", pattern=pattern["pattern"],
                           reason="sizing produced risk_per_share <= 0 or shares == 0")
            continue

        log.info(
            "SIZING %s: balance=%.2f risk$=%.2f entry=%.2f stop=%.2f risk/share=%.2f "
            "shares=%d target$=%.2f target=%.2f",
            symbol, state["balance"], sizing["risk_dollars"], sizing["entry"], sizing["stop"],
            sizing["risk_per_share"], sizing["shares"], sizing["target_dollars"], sizing["target"],
        )

        # Step 5 - final re-check right before placing
        state2 = get_account_state(trade_client, account_id)
        halt_reasons2 = check_halt(state2)
        if halt_reasons2 or len(state2["positions"]) >= MAX_OPEN_POSITIONS:
            reason = "; ".join(halt_reasons2) or f"{len(state2['positions'])} open positions >= max"
            notion_log_row(ticker=symbol, action="Halted", pattern=pattern["pattern"], reason=f"final check failed: {reason}")
            return

        result = place_limit_order(trade_client, account_id, symbol, sizing["shares"], sizing["entry"])
        action = "Taken" if result.get("placed") else "Skipped"
        reason = result.get("reason", "placed")
        if DRY_RUN and not result.get("placed"):
            reason = f"[DRY RUN] preview succeeded, order not actually placed ({reason})"
        notion_log_row(
            ticker=symbol, action=action,
            pattern=pattern["pattern"], entry=sizing["entry"], stop=sizing["stop"],
            target=sizing["target"], shares=sizing["shares"],
            reason=reason,
            outcome=json.dumps(result.get("response", {}))[:1900] if result.get("placed") else "",
        )
        took_trade = True
        break  # one trade per run; next 15-min run will re-scan

    if not took_trade:
        log.info("No trade taken this run (all candidates skipped -- see individual log rows).")


if __name__ == "__main__":
    main()
