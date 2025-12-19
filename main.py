import os
import json
import time
import math
import logging
from dataclasses import dataclass, asdict
from typing import Dict, Optional, List, Any, Tuple
from datetime import datetime, timezone

import ccxt
import pandas as pd
import numpy as np
from ta.momentum import RSIIndicator
from ta.trend import EMAIndicator
from telegram import Bot

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

    trade_usdt: float = 15.0
    max_open_positions: int = 5

    rsi_period: int = 14
    rsi_threshold: float = 45.0
    ema_period: int = 200
    ema_band_low: float = 0.98
    ema_band_high: float = 1.01

    take_profit_pct: float = 0.07
    stop_loss_pct: float = 0.08

    min_quote_volume_24h: float = 300_000.0
    top_symbols_limit: int = 300
    symbol_refresh_seconds: int = 15 * 60

    cycle_sleep_seconds: int = 60
    rate_limit_sleep_seconds: float = 0.12

    positions_file: str = "positions.json"
    history_file: str = "trade_history.json"
    equity_file: str = "equity_curve.json"


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
    target: float
    stop: float
    amount: float
    opened_at: float
    usdt_invested: float = 0.0


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
# Persistence
# ----------------------------


def load_positions(path: str) -> Dict[str, Position]:
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        out: Dict[str, Position] = {}
        for symbol, p in raw.items():
            out[symbol] = Position(**p)
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
# Telegram
# ----------------------------


class TelegramAlerter:
    def __init__(self, token: str, chat_id: str):
        self.bot = Bot(token=token)
        self.chat_id = chat_id
        self._last_error_sent_at = 0.0

    def send(self, msg: str) -> None:
        try:
            max_len = 4000
            if len(msg) <= max_len:
                self.bot.send_message(
                    chat_id=self.chat_id, text=msg, disable_web_page_preview=True)
            else:
                for i in range(0, len(msg), max_len):
                    chunk = msg[i:i+max_len]
                    self.bot.send_message(
                        chat_id=self.chat_id, text=chunk, disable_web_page_preview=True)
                    time.sleep(0.5)
        except Exception as e:
            logger.error(f"Erro Telegram: {e}")

    def send_error_throttled(self, msg: str, cooldown_seconds: int = 120) -> None:
        now = time.time()
        if now - self._last_error_sent_at < cooldown_seconds:
            return
        self._last_error_sent_at = now
        self.send(msg)

# ----------------------------
# Utils: retry/backoff
# ----------------------------


def is_retryable_exception(exc: Exception) -> bool:
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

    def price_to_precision(self, symbol: str, price: float) -> float:
        v = self.exchange.price_to_precision(symbol, price)
        return float(v)

# ----------------------------
# Strategy (usando biblioteca 'ta' em vez de TA-Lib)
# ----------------------------


def ohlcv_to_df(ohlcv: List[List[float]]) -> pd.DataFrame:
    df = pd.DataFrame(
        ohlcv, columns=["ts", "Open", "High", "Low", "Close", "Volume"])
    for c in ["Open", "High", "Low", "Close", "Volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df.dropna(inplace=True)
    return df


def compute_signal(df: pd.DataFrame, cfg: Config) -> Tuple[bool, float, float, float]:
    if len(df) < max(cfg.ema_period, cfg.rsi_period) + 5:
        return (False, 0.0, 0.0, 0.0)

    close_series = df["Close"]

    # Calcular RSI usando biblioteca 'ta'
    rsi_indicator = RSIIndicator(close=close_series, window=cfg.rsi_period)
    rsi = rsi_indicator.rsi().iloc[-1]

    # Calcular EMA usando biblioteca 'ta'
    ema_indicator = EMAIndicator(close=close_series, window=cfg.ema_period)
    ema = ema_indicator.ema_indicator().iloc[-1]

    curr_price = float(close_series.iloc[-1])

    # Verificar se está na banda e RSI abaixo do threshold
    in_band = (ema * cfg.ema_band_low) <= curr_price <= (ema * cfg.ema_band_high)
    entry_ok = (rsi < cfg.rsi_threshold) and in_band

    if not entry_ok:
        return (False, curr_price, 0.0, 0.0)

    target = curr_price * (1.0 + cfg.take_profit_pct)
    stop = curr_price * (1.0 - cfg.stop_loss_pct)
    return (True, curr_price, target, stop)

# ----------------------------
# Symbol selection
# ----------------------------


def get_top_symbols(client: BinanceClient, cfg: Config) -> List[str]:
    client.load_markets()
    tickers = client.fetch_tickers()

    blacklist = {"BUSD/USDT", "USDC/USDT", "DAI/USDT", "EUR/USDT", "GBP/USDT"}
    symbols: List[str] = []

    for s, data in tickers.items():
        if not s.endswith("/USDT"):
            continue
        if any(x in s for x in ["UP/", "DOWN/", "BEAR/", "BULL/"]):
            continue
        if s in blacklist:
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

    symbols_sorted = sorted(symbols, key=lambda x: float(
        tickers[x].get("quoteVolume") or 0.0), reverse=True)
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
        report += "📊 RELATÓRIO DE PERFORMANCE\n"
        report += "=" * 40 + "\n\n"

        report += f"Total de Operações: {stats['total_trades']}\n"
        report += f"✅ Ganhos: {stats['winning_trades']}\n"
        report += f"❌ Perdas: {stats['losing_trades']}\n"
        report += f"Taxa de Acerto: {stats['win_rate']:.2f}%\n\n"

        report += f"💰 P&L Total: ${stats['total_pnl']:.2f} USDT\n"
        report += f"📈 Média de Ganho: ${stats['avg_win']:.2f}\n"
        report += f"📉 Média de Perda: ${stats['avg_loss']:.2f}\n"
        report += f"🏆 Maior Ganho: ${stats['largest_win']:.2f}\n"
        report += f"💔 Maior Perda: ${stats['largest_loss']:.2f}\n"
        report += f"⚡ Fator de Lucro: {stats['profit_factor']:.2f}\n\n"

        if self.equity_curve:
            latest = self.equity_curve[-1]
            report += "=" * 40 + "\n"
            report += "💼 PATRIMÔNIO ATUAL\n"
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
                report += f"\n📊 Variação desde início: ${variation:.2f} ({variation_pct:+.2f}%)\n"

        if self.trades:
            report += "\n" + "=" * 40 + "\n"
            report += "📋 ÚLTIMAS 5 OPERAÇÕES\n"
            report += "=" * 40 + "\n"

            for trade in self.trades[-5:]:
                dt = datetime.fromtimestamp(trade.closed_at, tz=timezone.utc)
                emoji = "✅" if trade.pnl_usdt > 0 else "❌"
                report += f"\n{emoji} {trade.symbol}\n"
                report += f"   Entrada: ${trade.entry:.6f}\n"
                report += f"   Saída: ${trade.exit:.6f}\n"
                report += f"   P&L: ${trade.pnl_usdt:.2f} ({trade.pnl_pct:+.2f}%)\n"
                report += f"   Motivo: {trade.reason}\n"
                report += f"   Data: {dt.strftime('%Y-%m-%d %H:%M')} UTC\n"

        return report

# ----------------------------
# Trading engine
# ----------------------------


class TradingBot:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.client = BinanceClient(cfg)
        self.alerts = TelegramAlerter(cfg.tg_bot_token, cfg.tg_chat_id)
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

    def open_position(self, symbol: str, signal_price: float, target: float, stop: float) -> None:
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

            target_exec = avg_price * (1.0 + self.cfg.take_profit_pct)
            stop_exec = avg_price * (1.0 - self.cfg.stop_loss_pct)

            pos = Position(
                symbol=symbol,
                entry=avg_price,
                target=target_exec,
                stop=stop_exec,
                amount=amount,
                opened_at=time.time(),
                usdt_invested=self.cfg.trade_usdt
            )
            self.positions[symbol] = pos
            save_positions(self.cfg.positions_file, self.positions)

            self.alerts.send(
                "✅ COMPRA EXECUTADA\n"
                f"🚀 Ativo: {symbol}\n"
                f"💰 Entrada: ${avg_price:.6f}\n"
                f"🎯 Alvo: ${target_exec:.6f} (+{self.cfg.take_profit_pct*100:.1f}%)\n"
                f"🛑 Stop: ${stop_exec:.6f} (-{self.cfg.stop_loss_pct*100:.1f}%)\n"
                f"📦 Qtd: {amount:.8f}\n"
                f"💵 Investido: ${self.cfg.trade_usdt:.2f}"
            )
            logger.info(
                f"Posição aberta em {symbol} | entry={avg_price} amount={amount}")

        except Exception as e:
            logger.error(f"Erro ao comprar {symbol}: {e}")
            self.alerts.send_error_throttled(
                f"❌ ERRO COMPRA\nAtivo: {symbol}\nMotivo: {e}")

    def close_position(self, symbol: str, price: float, reason: str) -> None:
        pos = self.positions.get(symbol)
        if not pos:
            return
        try:
            amount_precise = self.client.amount_to_precision(
                symbol, pos.amount)
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

            emoji = "✅" if pnl_usdt > 0 else "❌"
            self.alerts.send(
                f"{emoji} VENDA EXECUTADA ({reason})\n"
                f"🚀 Ativo: {symbol}\n"
                f"💰 Entrada: ${pos.entry:.6f}\n"
                f"💸 Saída: ${exit_price:.6f}\n"
                f"📦 Qtd: {amount_precise:.8f}\n"
                f"\n💵 P&L: ${pnl_usdt:.2f} ({pnl_pct:+.2f}%)\n"
                f"💼 Investido: ${pos.usdt_invested:.2f}\n"
                f"💰 Retorno: ${proceeds:.2f}"
            )
            logger.info(
                f"Posição encerrada {symbol} | reason={reason} | price={exit_price} | pnl=${pnl_usdt:.2f}")

        except Exception as e:
            logger.error(f"Erro ao vender {symbol}: {e}")
            self.alerts.send_error_throttled(
                f"❌ ERRO VENDA\nAtivo: {symbol}\nMotivo: {e}")

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
                if last >= pos.target:
                    self.close_position(symbol, last, "TAKE PROFIT 🎯")
                elif last <= pos.stop:
                    self.close_position(symbol, last, "STOP LOSS 🛑")
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
                ok, curr_price, target, stop = compute_signal(df, self.cfg)
                if ok:
                    logger.info(f"Sinal em {s} | price={curr_price:.6f}")
                    self.open_position(s, curr_price, target, stop)
            except Exception as e:
                logger.warning(f"Erro na lógica {s}: {e}")

            time.sleep(self.cfg.rate_limit_sleep_seconds)

    def send_daily_report(self) -> None:
        report = self.performance.format_performance_report()
        self.alerts.send(report)

    def run_forever(self) -> None:
        logger.info("🤖 BOT INICIADO (BINANCE SPOT)")
        self.alerts.send("🤖 Bot iniciado e online!")

        self.send_daily_report()

        last_daily_report = time.time()

        while True:
            try:
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
