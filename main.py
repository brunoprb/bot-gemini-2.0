import os
import json
import time
import math
import logging
from dataclasses import dataclass, asdict
from typing import Dict, Optional, List, Any, Tuple
from datetime import datetime, timezone

import requests
import ccxt
import pandas as pd
import numpy as np
from ta.momentum import RSIIndicator
from ta.trend import EMAIndicator

# ----------------------------
# Logging
# ----------------------------
logger = logging.getLogger("crypto_bot")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

# ----------------------------
# Config
# ----------------------------


@dataclass(frozen=True)
class Config:
    binance_api_key: str
    binance_api_secret: str
    tg_bot_token: str
    tg_chat_id: str

    timeframe: str = "4h"
    ohlcv_limit: int = 300

    trade_usdt: float = 13.0  # valor por trade em USDT
    max_open_positions: int = 5

    rsi_period: int = 14
    rsi_threshold: float = 50.0  # mais frouxo

    ema_period: int = 200
    ema_band_low: float = 0.98
    ema_band_high: float = 1.01

    # Stop inicial (hard stop) enquanto trailing não arma
    stop_loss_pct: float = 0.08

    # Trailing stop (após armar)
    trailing_stop_pct: float = 0.04          # 4% abaixo do topo (peak)
    # arma trailing quando subir +2% desde entrada
    trailing_activate_pct: float = 0.02
    trailing_update_alert: bool = False      # se True, avisa quando stop subir

    min_quote_volume_24h: float = 300_000.0
    top_symbols_limit: int = 400             # monitora até 400 moedas
    symbol_refresh_seconds: int = 15 * 60

    cycle_sleep_seconds: int = 60
    rate_limit_sleep_seconds: float = 0.12

    positions_file: str = "positions.json"
    history_file: str = "trade_history.json"
    equity_file: str = "equity_curve.json"
    tg_offset_file: str = "tg_offset.json"


def load_config() -> Config:
    def must_get(name: str) -> str:
        v = os.getenv(name, "").strip()
        if not v:
            raise RuntimeError(f"Variável de ambiente ausente: {name}")
        return v

    return Config(
        binance_api_key=must_get("BINANCE_API_KEY"),
        binance_api_secret=must_get("BINANCE_API_SECRET"),
        tg_bot_token=must_get("TG_BOT_TOKEN"),
        tg_chat_id=must_get("TG_CHAT_ID"),
    )

# ----------------------------
# Data Models
# ----------------------------


@dataclass
class Position:
    symbol: str
    entry: float
    stop: float
    amount: float
    opened_at: float
    usdt_invested: float = 0.0

    peak: float = 0.0
    trailing_armed: bool = False


@dataclass
class TradeHistory:
    symbol: str
    entry: float
    exit: float
    amount: float
    opened_at: float
    closed_at: float
    reason: str
    pnl_usdt: float
    pnl_pct: float
    usdt_invested: float


@dataclass
class EquitySnapshot:
    timestamp: float
    total_equity_usdt: float
    free_balance_usdt: float
    positions_value_usdt: float
    open_positions: int
    total_trades: int
    winning_trades: int
    losing_trades: int

# ----------------------------
# Persistence helpers (tolerantes a mudanças de schema)
# ----------------------------


def _pick(d: Dict[str, Any], keys: List[str]) -> Dict[str, Any]:
    return {k: d[k] for k in keys if k in d}


def load_positions(path: str) -> Dict[str, Position]:
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            raw = json.load(f)

        out: Dict[str, Position] = {}
        for symbol, p in raw.items():
            allowed = _pick(p, [
                "symbol", "entry", "stop", "amount", "opened_at",
                "usdt_invested", "peak", "trailing_armed"
            ])
            allowed["symbol"] = allowed.get("symbol") or symbol
            out[symbol] = Position(**allowed)
        return out
    except Exception as e:
        logger.error(f"Falha ao carregar posições de {path}: {e}")
        return {}


def save_positions(path: str, positions: Dict[str, Position]) -> None:
    try:
        raw = {s: asdict(p) for s, p in positions.items()}
        with open(path, "w", encoding="utf-8") as f:
            json.dump(raw, f, ensure_ascii=False, indent=2)
    except Exception as e:
        logger.error(f"Falha ao salvar posições em {path}: {e}")


def load_trade_history(path: str) -> List[TradeHistory]:
    if not os.path.exists(path):
        return []
    try:
        with open(path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        return [TradeHistory(**t) for t in raw]
    except Exception as e:
        logger.error(f"Falha ao carregar histórico de {path}: {e}")
        return []


def save_trade_history(path: str, history: List[TradeHistory]) -> None:
    try:
        raw = [asdict(t) for t in history]
        with open(path, "w", encoding="utf-8") as f:
            json.dump(raw, f, ensure_ascii=False, indent=2)
    except Exception as e:
        logger.error(f"Falha ao salvar histórico em {path}: {e}")


def load_equity_curve(path: str) -> List[EquitySnapshot]:
    if not os.path.exists(path):
        return []
    try:
        with open(path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        return [EquitySnapshot(**e) for e in raw]
    except Exception as e:
        logger.error(f"Falha ao carregar curva de equity de {path}: {e}")
        return []


def save_equity_curve(path: str, equity: List[EquitySnapshot]) -> None:
    try:
        raw = [asdict(e) for e in equity]
        with open(path, "w", encoding="utf-8") as f:
            json.dump(raw, f, ensure_ascii=False, indent=2)
    except Exception as e:
        logger.error(f"Falha ao salvar curva de equity em {path}: {e}")

# ----------------------------
# Telegram (síncrono via HTTP)
# ----------------------------


class TelegramAlerter:
    def __init__(self, token: str, chat_id: str):
        self.token = token
        self.chat_id = str(chat_id)
        self._last_error_sent_at = 0.0

    def send(self, msg: str) -> None:
        try:
            url = f"https://api.telegram.org/bot{self.token}/sendMessage"
            payload = {
                "chat_id": self.chat_id,
                "text": msg,
                "disable_web_page_preview": True,
            }

            max_len = 3800
            if len(msg) <= max_len:
                r = requests.post(url, json=payload, timeout=20)
                r.raise_for_status()
                return

            for i in range(0, len(msg), max_len):
                chunk = msg[i:i + max_len]
                payload["text"] = chunk
                r = requests.post(url, json=payload, timeout=20)
                r.raise_for_status()
                time.sleep(0.4)

        except Exception as e:
            logger.error(f"Erro Telegram: {e}")

    def send_error_throttled(self, msg: str, cooldown_seconds: int = 120) -> None:
        now = time.time()
        if now - self._last_error_sent_at < cooldown_seconds:
            return
        self._last_error_sent_at = now
        self.send(msg)


class TelegramCommandPoller:
    def __init__(self, token: str, chat_id: str, state_file: str):
        self.token = token
        self.chat_id = str(chat_id)
        self.state_file = state_file
        self.offset = self._load_offset()

    def _load_offset(self) -> int:
        try:
            if os.path.exists(self.state_file):
                with open(self.state_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                return int(data.get("offset", 0))
        except Exception:
            pass
        return 0

    def _save_offset(self) -> None:
        try:
            with open(self.state_file, "w", encoding="utf-8") as f:
                json.dump({"offset": self.offset}, f)
        except Exception:
            pass

    def poll(self, handle_message_fn) -> None:
        try:
            url = f"https://api.telegram.org/bot{self.token}/getUpdates"
            params = {
                "timeout": 0,
                "offset": self.offset,
                "allowed_updates": ["message"],
            }
            r = requests.get(url, params=params, timeout=20)
            r.raise_for_status()
            data = r.json()
            updates = data.get("result", [])

            for upd in updates:
                upd_id = int(upd.get("update_id", 0))
                self.offset = max(self.offset, upd_id + 1)

                msg = upd.get("message") or {}
                chat = msg.get("chat") or {}
                chat_id = str(chat.get("id", ""))
                if chat_id != self.chat_id:
                    continue

                text = (msg.get("text") or "").strip()
                if text:
                    handle_message_fn(text)

            if updates:
                self._save_offset()

        except Exception as e:
            logger.warning(f"Falha ao poll Telegram updates: {e}")

# ----------------------------
# Utils: retry/backoff
# ----------------------------


def is_retryable_exception(exc: Exception) -> bool:
    if isinstance(exc, (ccxt.AuthenticationError, ccxt.PermissionDenied, ccxt.AccountSuspended)):
        return False

    retryable_types = (
        ccxt.NetworkError,
        ccxt.ExchangeNotAvailable,
        ccxt.RequestTimeout,
        ccxt.DDoSProtection,
        ccxt.RateLimitExceeded,
    )
    return isinstance(exc, retryable_types)


def call_with_retries(fn, *args, retries: int = 5, base_sleep: float = 0.7, **kwargs):
    last_exc = None
    for attempt in range(retries):
        try:
            return fn(*args, **kwargs)
        except Exception as e:
            last_exc = e
            if not is_retryable_exception(e):
                raise
            sleep_s = base_sleep * (2 ** attempt) + (0.05 * attempt)
            logger.warning(
                f"Retryable error: {e} | tentativa {attempt+1}/{retries} | aguardando {sleep_s:.2f}s")
            time.sleep(sleep_s)
    raise last_exc

# ----------------------------
# Exchange wrapper
# ----------------------------


class BinanceClient:
    def __init__(self, cfg: Config):
        self.exchange = ccxt.binance({
            "apiKey": cfg.binance_api_key,
            "secret": cfg.binance_api_secret,
            "enableRateLimit": True,
            "timeout": 30_000,
            "options": {
                "defaultType": "spot",
                "adjustForTimeDifference": True,
            }
        })
        self.markets_loaded = False

    def load_markets(self) -> None:
        if self.markets_loaded:
            return
        call_with_retries(self.exchange.load_markets)
        self.markets_loaded = True

    def fetch_balance(self) -> Dict[str, Any]:
        return call_with_retries(self.exchange.fetch_balance)

    def fetch_tickers(self) -> Dict[str, Any]:
        return call_with_retries(self.exchange.fetch_tickers)

    def fetch_ohlcv(self, symbol: str, timeframe: str, limit: int) -> List[List[float]]:
        return call_with_retries(self.exchange.fetch_ohlcv, symbol, timeframe, None, limit)

    def fetch_ticker(self, symbol: str) -> Dict[str, Any]:
        return call_with_retries(self.exchange.fetch_ticker, symbol)

    def create_market_buy_usdt(self, symbol: str, usdt_amount: float) -> Dict[str, Any]:
        params = {"quoteOrderQty": usdt_amount}
        return call_with_retries(self.exchange.create_market_buy_order, symbol, 0, params)

    def create_market_sell(self, symbol: str, amount_base: float) -> Dict[str, Any]:
        return call_with_retries(self.exchange.create_market_sell_order, symbol, amount_base)

    def market_info(self, symbol: str) -> Dict[str, Any]:
        self.load_markets()
        return self.exchange.market(symbol)

    def amount_to_precision(self, symbol: str, amount: float) -> float:
        v = self.exchange.amount_to_precision(symbol, amount)
        return float(v)

# ----------------------------
# Strategy
# ----------------------------


def ohlcv_to_df(ohlcv: List[List[float]]) -> pd.DataFrame:
    df = pd.DataFrame(
        ohlcv, columns=["ts", "Open", "High", "Low", "Close", "Volume"])
    for c in ["Open", "High", "Low", "Close", "Volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df.dropna(inplace=True)
    return df


def compute_signal(df: pd.DataFrame, cfg: Config) -> Tuple[bool, float, float]:
    if len(df) < max(cfg.ema_period, cfg.rsi_period) + 5:
        return (False, 0.0, 0.0)

    close_series = df["Close"]

    rsi_indicator = RSIIndicator(close=close_series, window=cfg.rsi_period)
    rsi = float(rsi_indicator.rsi().iloc[-1])

    ema_indicator = EMAIndicator(close=close_series, window=cfg.ema_period)
    ema = float(ema_indicator.ema_indicator().iloc[-1])

    curr_price = float(close_series.iloc[-1])

    in_band = (ema * cfg.ema_band_low) <= curr_price <= (ema * cfg.ema_band_high)
    entry_ok = (rsi < cfg.rsi_threshold) and in_band

    if not entry_ok:
        return (False, curr_price, 0.0)

    initial_stop = curr_price * (1.0 - cfg.stop_loss_pct)
    return (True, curr_price, initial_stop)

# ----------------------------
# Symbol selection
# ----------------------------


def get_top_symbols(client: BinanceClient, cfg: Config) -> List[str]:
    client.load_markets()
    tickers = client.fetch_tickers()

    stablecoin_blacklist = {
        "BUSD/USDT", "USDC/USDT", "DAI/USDT", "XUSD/USDT", "USD1/USDT", "BFUSD/USDT", "TUSD/USDT", "FDUSD/USDT",
        "EUR/USDT", "EURI/USDT", "GBP/USDT", "AUD/USDT", "USDE/USDT", "USDP/USDT",
        "PYUSD/USDT", "GUSD/USDT", "USDD/USDT", "USDN/USDT", "USDJ/USDT",
        "PAX/USDT", "HUSD/USDT", "SUSD/USDT", "CUSD/USDT", "FRAX/USDT",
        "USDX/USDT", "RSV/USDT", "VAI/USDT", "UST/USDT", "USTC/USDT",
        "ALUSD/USDT", "LUSD/USDT", "MIM/USDT", "FEI/USDT", "RAI/USDT",
        "DOLA/USDT", "BEAN/USDT", "OUSD/USDT", "FLEXUSD/USDT", "EURS/USDT"
    }

    symbols: List[str] = []

    for s, data in tickers.items():
        if not s.endswith("/USDT"):
            continue

        base = s.split("/")[0].upper()
        if base in ["EUR", "GBP", "AUD", "BUSD", "USDC", "DAI", "TUSD", "FDUSD", "USDE", "USDP", "PYUSD"]:
            continue

        if s in stablecoin_blacklist:
            continue

        if any(x in s for x in ["UP/", "DOWN/", "BEAR/", "BULL/"]):
            continue

        try:
            m = client.market_info(s)
            if not m.get("active", True):
                continue
            if m.get("spot") is False:
                continue
        except Exception:
            continue

        qv = data.get("quoteVolume") or 0.0
        try:
            qv = float(qv)
        except Exception:
            qv = 0.0

        if qv >= cfg.min_quote_volume_24h:
            symbols.append(s)

    symbols_sorted = sorted(
        symbols,
        key=lambda x: float(tickers[x].get("quoteVolume") or 0.0),
        reverse=True
    )
    return symbols_sorted[:cfg.top_symbols_limit]

# ----------------------------
# Performance Analytics
# ----------------------------


class PerformanceTracker:
    def __init__(self, history_path: str, equity_path: str):
        self.history_path = history_path
        self.equity_path = equity_path
        self.trades: List[TradeHistory] = load_trade_history(history_path)
        self.equity_curve: List[EquitySnapshot] = load_equity_curve(
            equity_path)

    def add_trade(self, trade: TradeHistory) -> None:
        self.trades.append(trade)
        save_trade_history(self.history_path, self.trades)

    def add_equity_snapshot(self, snapshot: EquitySnapshot) -> None:
        self.equity_curve.append(snapshot)
        save_equity_curve(self.equity_path, self.equity_curve)

    def get_stats(self) -> Dict[str, Any]:
        if not self.trades:
            return {
                "total_trades": 0,
                "winning_trades": 0,
                "losing_trades": 0,
                "win_rate": 0.0,
                "total_pnl": 0.0,
                "avg_win": 0.0,
                "avg_loss": 0.0,
                "largest_win": 0.0,
                "largest_loss": 0.0,
                "profit_factor": 0.0,
            }

        wins = [t for t in self.trades if t.pnl_usdt > 0]
        losses = [t for t in self.trades if t.pnl_usdt <= 0]

        total_pnl = sum(t.pnl_usdt for t in self.trades)
        total_win = sum(t.pnl_usdt for t in wins)
        total_loss = abs(sum(t.pnl_usdt for t in losses))

        return {
            "total_trades": len(self.trades),
            "winning_trades": len(wins),
            "losing_trades": len(losses),
            "win_rate": (len(wins) / len(self.trades) * 100) if self.trades else 0.0,
            "total_pnl": total_pnl,
            "avg_win": (total_win / len(wins)) if wins else 0.0,
            "avg_loss": (total_loss / len(losses)) if losses else 0.0,
            "largest_win": max((t.pnl_usdt for t in wins), default=0.0),
            "largest_loss": min((t.pnl_usdt for t in losses), default=0.0),
            "profit_factor": (total_win / total_loss) if total_loss > 0 else 0.0,
        }

    def format_performance_report(self) -> str:
        stats = self.get_stats()

        report = "=" * 40 + "\n"
        report += "RELATÓRIO DE PERFORMANCE\n"
        report += "=" * 40 + "\n\n"

        report += f"Total de Operações: {stats['total_trades']}\n"
        report += f"Ganhos: {stats['winning_trades']}\n"
        report += f"Perdas: {stats['losing_trades']}\n"
        report += f"Taxa de Acerto: {stats['win_rate']:.2f}%\n\n"

        report += f"P&L Total: ${stats['total_pnl']:.2f} USDT\n"
        report += f"Média de Ganho: ${stats['avg_win']:.2f}\n"
        report += f"Média de Perda: ${stats['avg_loss']:.2f}\n"
        report += f"Maior Ganho: ${stats['largest_win']:.2f}\n"
        report += f"Maior Perda: ${stats['largest_loss']:.2f}\n"
        report += f"Fator de Lucro: {stats['profit_factor']:.2f}\n\n"

        if self.equity_curve:
            latest = self.equity_curve[-1]
            report += "=" * 40 + "\n"
            report += "PATRIMÔNIO ATUAL\n"
            report += "=" * 40 + "\n"
            report += f"Equity Total: ${latest.total_equity_usdt:.2f}\n"
            report += f"Saldo Livre: ${latest.free_balance_usdt:.2f}\n"
            report += f"Em Posições: ${latest.positions_value_usdt:.2f}\n"
            report += f"Posições Abertas: {latest.open_positions}\n"

            if len(self.equity_curve) > 1:
                first = self.equity_curve[0]
                variation = latest.total_equity_usdt - first.total_equity_usdt
                variation_pct = (variation / first.total_equity_usdt *
                                 100) if first.total_equity_usdt > 0 else 0
                report += f"\nVariação desde início: ${variation:.2f} ({variation_pct:+.2f}%)\n"

        if self.trades:
            report += "\n" + "=" * 40 + "\n"
            report += "ÚLTIMAS 5 OPERAÇÕES\n"
            report += "=" * 40 + "\n"

            for trade in self.trades[-5:]:
                dt = datetime.fromtimestamp(trade.closed_at, tz=timezone.utc)
                tag = "WIN" if trade.pnl_usdt > 0 else "LOSS"
                report += f"\n[{tag}] {trade.symbol}\n"
                report += f"  Entrada: ${trade.entry:.6f}\n"
                report += f"  Saída:   ${trade.exit:.6f}\n"
                report += f"  P&L:     ${trade.pnl_usdt:.2f} ({trade.pnl_pct:+.2f}%)\n"
                report += f"  Motivo:  {trade.reason}\n"
                report += f"  Data:    {dt.strftime('%Y-%m-%d %H:%M')} UTC\n"

        return report

# ----------------------------
# Trading engine
# ----------------------------


class TradingBot:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.client = BinanceClient(cfg)
        self.alerts = TelegramAlerter(cfg.tg_bot_token, cfg.tg_chat_id)
        self.cmd = TelegramCommandPoller(
            cfg.tg_bot_token, cfg.tg_chat_id, cfg.tg_offset_file)

        self.positions: Dict[str, Position] = load_positions(
            cfg.positions_file)
        self.performance = PerformanceTracker(
            cfg.history_file, cfg.equity_file)

        self._symbols_cache: List[str] = []
        self._symbols_cache_at: float = 0.0
        self._last_equity_snapshot: float = 0.0

    def refresh_symbols_if_needed(self) -> List[str]:
        now = time.time()
        if self._symbols_cache and (now - self._symbols_cache_at) < self.cfg.symbol_refresh_seconds:
            return self._symbols_cache

        syms = get_top_symbols(self.client, self.cfg)
        self._symbols_cache = syms
        self._symbols_cache_at = now
        logger.info(f"Lista de símbolos atualizada: {len(syms)} ativos")
        return syms

    def can_open_new_position(self) -> bool:
        return len(self.positions) < self.cfg.max_open_positions

    def fetch_df(self, symbol: str) -> Optional[pd.DataFrame]:
        try:
            ohlcv = self.client.fetch_ohlcv(
                symbol, self.cfg.timeframe, self.cfg.ohlcv_limit)
            df = ohlcv_to_df(ohlcv)
            return df
        except Exception as e:
            logger.warning(f"Falha ao buscar OHLCV {symbol}: {e}")
            return None

    def calculate_positions_value(self) -> float:
        total = 0.0
        for symbol, pos in self.positions.items():
            try:
                ticker = self.client.fetch_ticker(symbol)
                last_price = float(ticker.get("last") or 0.0)
                total += pos.amount * last_price
            except Exception as e:
                logger.warning(f"Erro ao calcular valor de {symbol}: {e}")
        return total

    def take_equity_snapshot(self) -> None:
        now = time.time()
        if now - self._last_equity_snapshot < 14400:
            return

        try:
            balance = self.client.fetch_balance()
            free_usdt = float(balance.get("USDT", {}).get("free", 0.0))
            positions_value = self.calculate_positions_value()
            total_equity = free_usdt + positions_value

            stats = self.performance.get_stats()

            snapshot = EquitySnapshot(
                timestamp=now,
                total_equity_usdt=total_equity,
                free_balance_usdt=free_usdt,
                positions_value_usdt=positions_value,
                open_positions=len(self.positions),
                total_trades=stats["total_trades"],
                winning_trades=stats["winning_trades"],
                losing_trades=stats["losing_trades"],
            )

            self.performance.add_equity_snapshot(snapshot)
            self._last_equity_snapshot = now
            logger.info(f"Snapshot de equity: ${total_equity:.2f} USDT")

        except Exception as e:
            logger.error(f"Erro ao tirar snapshot de equity: {e}")

    def sync_positions_with_wallet(self) -> None:
        """
        Garante que não vamos contar como posição aberta algo que já foi
        totalmente vendido manualmente (só sobrou USDT).
        """
        if not self.positions:
            return
        try:
            balance = self.client.fetch_balance()
        except Exception as e:
            logger.warning(
                f"Não foi possível sincronizar posições com saldo: {e}")
            return

        removed: List[str] = []
        for symbol in list(self.positions.keys()):
            base = symbol.split("/")[0]
            free = float(balance.get(base, {}).get("free", 0.0))
            if free <= 0:
                removed.append(symbol)
                del self.positions[symbol]

        if removed:
            save_positions(self.cfg.positions_file, self.positions)
            logger.info(
                f"Removidas posições sem saldo real: {', '.join(removed)}")

    def open_position(self, symbol: str, signal_price: float) -> None:
        if symbol in self.positions:
            return
        if not self.can_open_new_position():
            return

        try:
            market = self.client.market_info(symbol)

            min_cost = None
            limits = (market or {}).get("limits") or {}
            cost_limits = limits.get("cost") or {}
            if cost_limits.get("min") is not None:
                try:
                    min_cost = float(cost_limits["min"])
                except Exception:
                    min_cost = None

            if min_cost is not None and self.cfg.trade_usdt < min_cost:
                logger.info(
                    f"{symbol}: trade_usdt {self.cfg.trade_usdt} abaixo do minNotional {min_cost}. Pulando.")
                return

            order = self.client.create_market_buy_usdt(
                symbol, self.cfg.trade_usdt)

            amount = float(order.get("amount") or 0.0)
            avg_price = order.get("average")
            if avg_price is None:
                avg_price = self.client.fetch_ticker(
                    symbol).get("last") or signal_price
            avg_price = float(avg_price)

            stop_exec = avg_price * (1.0 - self.cfg.stop_loss_pct)

            pos = Position(
                symbol=symbol,
                entry=avg_price,
                stop=stop_exec,
                amount=amount,
                opened_at=time.time(),
                usdt_invested=self.cfg.trade_usdt,
                peak=avg_price,
                trailing_armed=False,
            )
            self.positions[symbol] = pos
            save_positions(self.cfg.positions_file, self.positions)

            self.alerts.send(
                "COMPRA EXECUTADA\n"
                f"Ativo: {symbol}\n"
                f"Entrada: ${avg_price:.6f}\n"
                f"Stop inicial: ${stop_exec:.6f} (-{self.cfg.stop_loss_pct*100:.1f}%)\n"
                f"Trailing: {self.cfg.trailing_stop_pct*100:.1f}% (arma após +{self.cfg.trailing_activate_pct*100:.1f}%)\n"
                f"Qtd: {amount:.8f}\n"
                f"Investido: ${self.cfg.trade_usdt:.2f}"
            )
            logger.info(
                f"Posição aberta em {symbol} | entry={avg_price} amount={amount}")

        except Exception as e:
            logger.error(f"Erro ao comprar {symbol}: {e}")
            self.alerts.send_error_throttled(
                f"ERRO COMPRA\nAtivo: {symbol}\nMotivo: {e}")

    def close_position(self, symbol: str, price: float, reason: str) -> None:
        """
        Usa o saldo REAL da carteira para vender, evitando erro de
        'insufficient balance' em moedas tipo LUNC.
        """
        pos = self.positions.get(symbol)
        if not pos:
            return
        try:
            base_asset = symbol.split("/")[0]
            balance = self.client.fetch_balance()
            free_amount = float(balance.get(base_asset, {}).get("free", 0.0))

            if free_amount <= 0:
                logger.warning(
                    f"Saldo livre de {base_asset} é zero, abortando venda de {symbol}.")
                return

            amount_precise = self.client.amount_to_precision(
                symbol, free_amount)
            if amount_precise <= 0:
                raise RuntimeError(
                    f"Quantidade inválida para vender: {amount_precise}")

            order = self.client.create_market_sell(symbol, amount_precise)

            exit_price = order.get("average")
            if exit_price is None:
                exit_price = price
            exit_price = float(exit_price)

            proceeds = amount_precise * exit_price
            pnl_usdt = proceeds - pos.usdt_invested
            pnl_pct = (pnl_usdt / pos.usdt_invested *
                       100) if pos.usdt_invested > 0 else 0.0

            trade = TradeHistory(
                symbol=symbol,
                entry=pos.entry,
                exit=exit_price,
                amount=amount_precise,
                opened_at=pos.opened_at,
                closed_at=time.time(),
                reason=reason,
                pnl_usdt=pnl_usdt,
                pnl_pct=pnl_pct,
                usdt_invested=pos.usdt_invested
            )
            self.performance.add_trade(trade)

            del self.positions[symbol]
            save_positions(self.cfg.positions_file, self.positions)

            tag = "WIN" if pnl_usdt > 0 else "LOSS"
            self.alerts.send(
                f"VENDA EXECUTADA ({tag})\n"
                f"Motivo: {reason}\n"
                f"Ativo: {symbol}\n"
                f"Entrada: ${pos.entry:.6f}\n"
                f"Saída:   ${exit_price:.6f}\n"
                f"Qtd: {amount_precise:.8f}\n"
                f"Peak: ${pos.peak:.6f}\n"
                f"Stop final: ${pos.stop:.6f}\n"
                f"\nP&L: ${pnl_usdt:.2f} ({pnl_pct:+.2f}%)\n"
                f"Investido: ${pos.usdt_invested:.2f}\n"
                f"Retorno: ${proceeds:.2f}"
            )
            logger.info(
                f"Posição encerrada {symbol} | reason={reason} | price={exit_price} | pnl=${pnl_usdt:.2f}")

        except Exception as e:
            logger.error(f"Erro ao vender {symbol}: {e}")
            self.alerts.send_error_throttled(
                f"ERRO VENDA\nAtivo: {symbol}\nMotivo: {e}")

    def monitor_exits(self) -> None:
        if not self.positions:
            return

        for symbol in list(self.positions.keys()):
            try:
                ticker = self.client.fetch_ticker(symbol)
                last = float(ticker.get("last") or 0.0)
                if last <= 0:
                    continue

                pos = self.positions[symbol]
                changed = False

                base_peak = pos.peak if pos.peak > 0 else pos.entry
                if last > base_peak:
                    pos.peak = last
                    changed = True

                arm_price = pos.entry * (1.0 + self.cfg.trailing_activate_pct)
                if (not pos.trailing_armed) and (last >= arm_price):
                    pos.trailing_armed = True
                    changed = True

                if pos.trailing_armed:
                    trail_stop = pos.peak * (1.0 - self.cfg.trailing_stop_pct)
                    if trail_stop > pos.stop:
                        pos.stop = trail_stop
                        changed = True
                        if self.cfg.trailing_update_alert:
                            self.alerts.send(
                                "TRAILING STOP ATUALIZADO\n"
                                f"Ativo: {symbol}\n"
                                f"Peak: {pos.peak:.6f}\n"
                                f"Novo Stop: {pos.stop:.6f}"
                            )

                if changed:
                    self.positions[symbol] = pos
                    save_positions(self.cfg.positions_file, self.positions)

                if last <= pos.stop:
                    reason = "TRAILING STOP" if pos.trailing_armed else "STOP LOSS"
                    self.close_position(symbol, last, reason)

            except Exception as e:
                logger.warning(f"Falha monitorando saída {symbol}: {e}")
                continue

    def scan_entries(self, symbols: List[str]) -> None:
        for s in symbols:
            if s in self.positions:
                continue
            if not self.can_open_new_position():
                break

            df = self.fetch_df(s)
            if df is None or len(df) < 50:
                time.sleep(self.cfg.rate_limit_sleep_seconds)
                continue

            try:
                ok, curr_price, _initial_stop = compute_signal(df, self.cfg)
                if ok:
                    logger.info(f"Sinal em {s} | price={curr_price:.6f}")
                    self.open_position(s, curr_price)
            except Exception as e:
                logger.warning(f"Erro na lógica {s}: {e}")

            time.sleep(self.cfg.rate_limit_sleep_seconds)

    def format_status(self) -> str:
        stats = self.performance.get_stats()
        lines = []
        lines.append("STATUS DO BOT")
        lines.append(
            f"Posições abertas: {len(self.positions)}/{self.cfg.max_open_positions}")
        lines.append(f"Trade size: ${self.cfg.trade_usdt}")
        lines.append(
            f"RSI<{self.cfg.rsi_threshold} | EMA{self.cfg.ema_period} band: {self.cfg.ema_band_low:.2f}-{self.cfg.ema_band_high:.2f}")
        lines.append(
            f"Trailing: {self.cfg.trailing_stop_pct*100:.1f}% | arma após +{self.cfg.trailing_activate_pct*100:.1f}%")
        lines.append(
            f"Trades: {stats['total_trades']} | WinRate: {stats['win_rate']:.2f}% | PnL total: ${stats['total_pnl']:.2f}")
        lines.append("")

        if not self.positions:
            lines.append("Nenhuma posição aberta no momento.")
            return "\n".join(lines)

        lines.append("POSIÇÕES:")
        for s, p in self.positions.items():
            try:
                last = float(self.client.fetch_ticker(s).get("last") or 0.0)
            except Exception:
                last = 0.0

            pnl = (p.amount * last - p.usdt_invested) if last > 0 else 0.0
            pnl_pct = (pnl / p.usdt_invested *
                       100) if p.usdt_invested > 0 and last > 0 else 0.0
            armed = "SIM" if p.trailing_armed else "NÃO"
            arm_price = p.entry * (1.0 + self.cfg.trailing_activate_pct)

            lines.append(
                f"- {s}\n"
                f"  Entry: {p.entry:.6f} | Last: {last:.6f}\n"
                f"  Peak: {p.peak:.6f} | Stop: {p.stop:.6f}\n"
                f"  Trailing armado: {armed} | Preço p/ armar: {arm_price:.6f}\n"
                f"  PnL est.: ${pnl:.2f} ({pnl_pct:+.2f}%)"
            )

        return "\n".join(lines)

    def handle_telegram_command(self, text: str) -> None:
        cmd = text.split()[0].lower()

        if cmd in ("/status", "status"):
            self.alerts.send(self.format_status())
            return

        if cmd in ("/report", "report"):
            self.alerts.send(self.performance.format_performance_report())
            return

        if cmd in ("/help", "help"):
            self.alerts.send(
                "Comandos:\n/status - resumo rápido\n/report - relatório completo\n/help - ajuda")
            return

    def send_daily_report(self) -> None:
        self.alerts.send(self.performance.format_performance_report())

    def run_forever(self) -> None:
        logger.info("🤖 BOT INICIADO (BINANCE SPOT)")
        self.alerts.send("🤖 Bot iniciado e online!")

        self.send_daily_report()
        last_daily_report = time.time()

        while True:
            try:
                # Sincroniza posições com o saldo real (se você tiver só USDT,
                # não conta mais como posição aberta)
                self.sync_positions_with_wallet()

                # Comandos Telegram
                self.cmd.poll(self.handle_telegram_command)

                symbols = self.refresh_symbols_if_needed()
                logger.info(
                    f"🔍 Varrendo {len(symbols)} ativos | posições abertas: {len(self.positions)}")

                self.scan_entries(symbols)
                self.monitor_exits()
                self.take_equity_snapshot()

                if time.time() - last_daily_report > 86400:
                    self.send_daily_report()
                    last_daily_report = time.time()

                logger.info(
                    f"✅ Ciclo concluído | posições ativas: {len(self.positions)}")
                time.sleep(self.cfg.cycle_sleep_seconds)

            except KeyboardInterrupt:
                logger.info("Encerrando por KeyboardInterrupt")
                self.alerts.send("🛑 Bot desligado (KeyboardInterrupt).")
                break
            except Exception as e:
                logger.error(f"Erro no loop principal: {e}")
                self.alerts.send_error_throttled(
                    f"⚠️ ERRO LOOP PRINCIPAL\nMotivo: {e}")
                time.sleep(10)


def main():
    cfg = load_config()
    bot = TradingBot(cfg)
    bot.run_forever()


if __name__ == "__main__":
    main()
