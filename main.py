# region imports
from AlgorithmImports import *
from collections import deque
from datetime import timedelta, time
# endregion


class CVDAbsorptionHarvester(QCAlgorithm):
    """
    ============================================================================
    CVD ABSORPTION HARVESTER  --  XFA V2.0
    ============================================================================
    Target account : Topstep 50K Express Funded Account (ES / MES)
    Instrument     : /ES  (E-mini S&P 500), 1 contract, $50 per point, 0.25 tick
    Data           : Tick resolution (trade ticks) + 1-minute consolidated bars

    THESIS
    ------
    Price makes a new 15-minute extreme, but the aggressive order flow behind
    that extreme *fails to follow*. That is absorption: a passive participant
    is eating the aggressive flow at the extreme. We fade the exhausted side.

      * Bid Absorption (LONG)  : new 15-bar LOW  + CVD higher than it was at
                                 that low  -> sellers pressed, price did not
                                 break, bids absorbed them.
      * Ask Absorption (SHORT) : new 15-bar HIGH + CVD lower than it was at
                                 that high -> buyers pressed, price did not
                                 break, offers absorbed them.

    RISK (hard-coded Topstep XFA governor, evaluated tick-by-tick)
    --------------------------------------------------------------
      * Zero-buffer killswitch : daily net PnL <= -$250  -> flatten + halt day
      * Winning day lock       : daily net PnL >= +$160  -> flatten + halt day
      * Session window         : entries only 09:45 - 11:30 ET
      * Hard session cutoff    : 11:30 ET -> flatten, ignore all further data
      * Per trade              : risk 2.00 pts ($100) / target 4.00 pts ($200)
      * Break-even trail       : at +2.00 pts favourable, stop -> entry +/- 0.25

    IMPLEMENTATION NOTES (read these before editing)
    ------------------------------------------------
      1. self.symbol is the CANONICAL (continuous) futures symbol. It is a
         DATA-ONLY symbol -- LEAN will reject orders sent to it. Every order
         is routed to self.contract, which tracks self.future.mapped and is
         refreshed on every rollover (SymbolChangedEvents).
      2. LEAN has no native OCO. The stop and the limit are two independent
         tickets; on_order_event cancels the survivor when either one fills.
      3. UpdateOrderFields is the *request object* passed to ticket.update().
         There is no ticket.update_order_fields() method in LEAN.
      4. DataNormalizationMode.Raw is used so that the prices in the
         consolidated signal bars are the same scale as the fill prices on
         the mapped contract. Backwards-ratio adjustment would desync them.
    ============================================================================
    """

    # ----------------------------------------------------------------- CONFIG
    TICK_SIZE            = 0.25      # ES minimum price increment
    POINT_VALUE          = 50.0      # $ per index point, 1 ES contract
    POSITION_SIZE        = 1         # contracts per trade

    TARGET_POINTS        = 4.00      # +$200 per contract
    STOP_POINTS          = 2.00      # -$100 per contract
    BE_TRIGGER_POINTS    = 2.00      # MFE required to arm break-even
    BE_OFFSET_POINTS     = 0.25      # one tick beyond entry once armed

    LOOKBACK             = 15        # divergence lookback, in 1-min bars

    DAILY_LOSS_LIMIT     = -250.0    # zero-buffer killswitch ($)
    DAILY_PROFIT_LOCK    = 160.0     # winning-day lock ($)

    SESSION_OPEN         = time(9, 30)   # daily state reset
    ENTRY_START          = time(9, 45)   # earliest new entry
    ENTRY_END            = time(11, 30)  # latest new entry / hard flatten

    # =========================================================================
    # 1. INITIALIZATION
    # =========================================================================
    def initialize(self):
        # ---- Backtest window & account -------------------------------------
        self.set_start_date(2024, 1, 2)
        self.set_end_date(2024, 3, 28)
        self.set_cash(50000)

        # All self.time comparisons below are therefore in US/Eastern.
        self.set_time_zone("America/New_York")

        # Futures-capable brokerage model (fees, margin, order support).
        self.set_brokerage_model(BrokerageName.InteractiveBrokersBrokerage,
                                 AccountType.Margin)

        # ---- Asset ---------------------------------------------------------
        # Continuous ES, rolled on open interest. Raw normalization keeps
        # signal prices and fill prices on the same scale.
        self.future = self.add_future(
            Futures.Indices.SP_500_E_MINI,
            resolution=Resolution.Tick,
            data_mapping_mode=DataMappingMode.OpenInterest,
            data_normalization_mode=DataNormalizationMode.Raw,
            contract_depth_offset=0,
        )

        # Canonical symbol: data subscription + consolidation ONLY.
        self.symbol = self.future.Symbol
        # Mapped symbol: the actual tradable contract. Set in on_data.
        self.contract = None

        # ---- Order flow state ----------------------------------------------
        self.cvd = 0                 # cumulative volume delta (session)
        self.last_tick_price = None  # previous trade print, for the tick rule

        # ---- Risk governor state -------------------------------------------
        self.halted_for_day = False
        self.start_of_day_portfolio_value = self.portfolio.total_portfolio_value

        # ---- Divergence FSM state ------------------------------------------
        # Each entry: (high, low, cvd_at_bar_close)
        self.bar_window = deque(maxlen=self.LOOKBACK)

        # ---- Live trade state ----------------------------------------------
        self.entry_ticket = None
        self.stop_ticket = None
        self.limit_ticket = None
        self.entry_price = None
        self.position_direction = 0   # +1 long, -1 short, 0 flat
        self.breakeven_armed = False

        # ---- Consolidator ---------------------------------------------------
        # NOTE: on a TICK subscription the TickType argument is REQUIRED,
        # otherwise LEAN cannot tell whether to aggregate trades or quotes.
        self.consolidate(self.symbol,
                         timedelta(minutes=1),
                         TickType.Trade,
                         self.on_minute_bar)

        # ---- Daily reset scheduler ------------------------------------------
        self.schedule.on(
            self.date_rules.every_day(self.symbol),
            self.time_rules.at(self.SESSION_OPEN.hour, self.SESSION_OPEN.minute),
            self.on_session_start,
        )

        self.set_warm_up(timedelta(0))

    # =========================================================================
    # 2. DAILY RESET  (09:30 ET)
    # =========================================================================
    def on_session_start(self):
        """Fresh session: clear the governor, the CVD, and the lookback."""
        # Defensive flatten -- nothing should ever be carried overnight.
        if self.portfolio.invested:
            self.liquidate(tag="SESSION_RESET_FLATTEN")
        self.transactions.cancel_open_orders(tag="SESSION_RESET")
        self._clear_trade_state()

        self.halted_for_day = False
        self.cvd = 0
        self.last_tick_price = None
        self.bar_window.clear()
        self.start_of_day_portfolio_value = self.portfolio.total_portfolio_value

        self.log(f"SESSION_START | equity={self.start_of_day_portfolio_value:.2f}")

    # =========================================================================
    # 3. TICK ENGINE + RISK GOVERNOR + BREAK-EVEN TRAIL
    # =========================================================================
    def on_data(self, data: Slice):
        # ---- 3a. Rollover: canonical -> new mapped contract -----------------
        if data.symbol_changed_events.count > 0:
            for changed in data.symbol_changed_events.values():
                if changed.symbol != self.symbol:
                    continue
                # Never hold a position through a roll.
                if self.portfolio.invested:
                    self.liquidate(tag="ROLLOVER_FLATTEN")
                self.transactions.cancel_open_orders(tag="ROLLOVER")
                self._clear_trade_state()
                self.last_tick_price = None
                self.log(f"ROLLOVER | {changed.old_symbol} -> {changed.new_symbol}")

        # Keep the tradable contract pointer current on every slice.
        mapped = self.future.mapped
        if mapped is not None:
            self.contract = mapped

        # ---- 3b. Order flow: cumulative volume delta ------------------------
        # Trade ticks arrive on the canonical symbol for a continuous future.
        # Fall back to the mapped contract so we never silently record zero.
        tick_list = None
        if self.symbol in data.ticks:
            tick_list = data.ticks[self.symbol]
        elif self.contract is not None and self.contract in data.ticks:
            tick_list = data.ticks[self.contract]

        if tick_list is not None:
            for tick in tick_list:
                # Trade prints only -- quotes carry no delta information.
                if tick.tick_type != TickType.Trade:
                    continue
                if tick.suspicious or tick.quantity <= 0:
                    continue

                price = tick.price
                if self.last_tick_price is not None:
                    if price > self.last_tick_price:
                        # Uptick -> lifted the offer -> aggressive buyer.
                        self.cvd += tick.quantity
                    elif price < self.last_tick_price:
                        # Downtick -> hit the bid -> aggressive seller.
                        self.cvd -= tick.quantity
                    # Flat prints contribute nothing under the strict tick rule.
                self.last_tick_price = price

        # ---- 3c. Topstep XFA governor (tick-by-tick) ------------------------
        if self.halted_for_day:
            return

        daily_pnl = (self.portfolio.total_portfolio_value
                     - self.start_of_day_portfolio_value)

        # Zero-buffer killswitch.
        if daily_pnl <= self.DAILY_LOSS_LIMIT:
            self._halt_day("HALTED_FAIL", daily_pnl)
            return

        # Winning-day lock.
        if daily_pnl >= self.DAILY_PROFIT_LOCK:
            self._halt_day("HALTED_SUCCESS", daily_pnl)
            return

        # Hard session cutoff -- flatten and ignore everything after 11:30 ET.
        if self.time.time() >= self.ENTRY_END:
            if self.portfolio.invested or self._has_working_orders():
                self.liquidate(tag="SESSION_CUTOFF")
                self.transactions.cancel_open_orders(tag="SESSION_CUTOFF")
                self._clear_trade_state()
                self.log(f"SESSION_CUTOFF | daily_pnl={daily_pnl:.2f}")
            return

        # ---- 3d. Break-even trail on the live position ----------------------
        self._manage_open_position()

    # =========================================================================
    # 4. BREAK-EVEN TRAILING
    # =========================================================================
    def _manage_open_position(self):
        """At +2.00 pts MFE, slide the stop to entry +/- one tick."""
        if self.breakeven_armed:
            return
        if self.contract is None or self.entry_price is None:
            return
        if self.stop_ticket is None or not self._is_ticket_working(self.stop_ticket):
            return
        if not self.portfolio[self.contract].invested:
            return

        security = self.securities[self.contract]
        price = security.price
        if price == 0:
            return

        # Maximum favourable excursion, in points.
        mfe = (price - self.entry_price) * self.position_direction
        if mfe < self.BE_TRIGGER_POINTS:
            return

        new_stop = self._round_to_tick(
            self.entry_price + (self.BE_OFFSET_POINTS * self.position_direction)
        )

        update_fields = UpdateOrderFields()
        update_fields.stop_price = new_stop
        update_fields.tag = "XFA_BREAKEVEN"
        response = self.stop_ticket.update(update_fields)

        if response.is_success:
            self.breakeven_armed = True
            self.log(f"BREAKEVEN | entry={self.entry_price:.2f} "
                     f"stop->{new_stop:.2f} mfe={mfe:.2f}")

    # =========================================================================
    # 5. DIVERGENCE FSM  (1-minute bars)
    # =========================================================================
    def on_minute_bar(self, bar: TradeBar):
        """
        Absorption divergence against a 15-bar lookback.

        The lookback window holds the PREVIOUS 15 bars and, critically, the
        exact self.cvd value at each of those bar closes. The current bar is
        evaluated against that window and only then appended.
        """
        # Snapshot CVD at this bar's close before anything else touches it.
        current_cvd = self.cvd

        if len(self.bar_window) == self.LOOKBACK and self._can_enter():
            highs = [b[0] for b in self.bar_window]
            lows = [b[1] for b in self.bar_window]

            lowest_low = min(lows)
            highest_high = max(highs)

            # Most recent occurrence of each extreme is the reference pivot.
            low_idx = len(lows) - 1 - lows[::-1].index(lowest_low)
            high_idx = len(highs) - 1 - highs[::-1].index(highest_high)

            cvd_at_low = self.bar_window[low_idx][2]
            cvd_at_high = self.bar_window[high_idx][2]

            is_green = bar.close > bar.open
            is_red = bar.close < bar.open

            # ---- LONG: bid absorption --------------------------------------
            # New low swept, bar closes green, and net aggression is HIGHER
            # than it was at the prior low -> sellers were absorbed.
            if is_green and bar.low <= lowest_low and current_cvd > cvd_at_low:
                self._enter(
                    direction=1,
                    reason=(f"BID_ABSORPTION low={bar.low:.2f}<=({lowest_low:.2f}) "
                            f"cvd={current_cvd}>({cvd_at_low})")
                )

            # ---- SHORT: ask absorption -------------------------------------
            # New high swept, bar closes red, and net aggression is LOWER
            # than it was at the prior high -> buyers were absorbed.
            elif is_red and bar.high >= highest_high and current_cvd < cvd_at_high:
                self._enter(
                    direction=-1,
                    reason=(f"ASK_ABSORPTION high={bar.high:.2f}>=({highest_high:.2f}) "
                            f"cvd={current_cvd}<({cvd_at_high})")
                )

        # Roll the window forward.
        self.bar_window.append((bar.high, bar.low, current_cvd))

    def _can_enter(self) -> bool:
        """All gates that must be open before a new position is allowed."""
        if self.halted_for_day:
            return False
        if self.contract is None:
            return False
        if self.is_warming_up:
            return False

        now = self.time.time()
        if now < self.ENTRY_START or now >= self.ENTRY_END:
            return False

        # One position at a time, and never stack on top of working orders.
        if self.portfolio.invested:
            return False
        if self._has_working_orders():
            return False
        if self.securities[self.contract].price == 0:
            return False

        return True

    # =========================================================================
    # 6. EXECUTION  (market entry + OCO bracket)
    # =========================================================================
    def _enter(self, direction: int, reason: str):
        quantity = self.POSITION_SIZE * direction
        self.position_direction = direction
        self.breakeven_armed = False

        self.entry_ticket = self.market_order(self.contract, quantity,
                                              tag="XFA_ENTRY")

        side = "LONG" if direction > 0 else "SHORT"
        self.log(f"ENTRY {side} | {self.contract.value} | {reason}")

        # In backtesting a market order normally fills synchronously, so the
        # ticket already carries the fill price. If it does not (live, or a
        # partial), on_order_event places the bracket instead.
        if self.entry_ticket.status == OrderStatus.Filled:
            self._place_brackets(self.entry_ticket.average_fill_price,
                                 self.entry_ticket.quantity_filled)

    def _place_brackets(self, fill_price: float, filled_quantity: float):
        """Attach the 2-point stop and the 4-point target to a filled entry."""
        # Idempotent: on_order_event and _enter may both reach this.
        if self.stop_ticket is not None or self.limit_ticket is not None:
            return
        if filled_quantity == 0 or self.contract is None:
            return

        direction = 1 if filled_quantity > 0 else -1
        self.entry_price = fill_price
        self.position_direction = direction

        stop_price = self._round_to_tick(
            fill_price - (self.STOP_POINTS * direction))
        limit_price = self._round_to_tick(
            fill_price + (self.TARGET_POINTS * direction))

        exit_quantity = -filled_quantity  # flattening side

        self.stop_ticket = self.stop_market_order(
            self.contract, exit_quantity, stop_price, tag="XFA_STOP")
        self.limit_ticket = self.limit_order(
            self.contract, exit_quantity, limit_price, tag="XFA_TARGET")

        self.log(f"BRACKET | fill={fill_price:.2f} stop={stop_price:.2f} "
                 f"target={limit_price:.2f} qty={filled_quantity}")

    # =========================================================================
    # 7. ORDER EVENTS  (manual OCO)
    # =========================================================================
    def on_order_event(self, order_event: OrderEvent):
        if order_event.status != OrderStatus.Filled:
            return

        order_id = order_event.order_id

        # ---- Entry filled -> attach the bracket -----------------------------
        if self.entry_ticket is not None and order_id == self.entry_ticket.order_id:
            self._place_brackets(order_event.fill_price, order_event.fill_quantity)
            return

        # ---- Stop filled -> cancel the target (OCO leg 1) -------------------
        if self.stop_ticket is not None and order_id == self.stop_ticket.order_id:
            if self.limit_ticket is not None and self._is_ticket_working(self.limit_ticket):
                self.limit_ticket.cancel("OCO_STOP_FILLED")
            self.log(f"EXIT_STOP | fill={order_event.fill_price:.2f} "
                     f"be_armed={self.breakeven_armed}")
            self._clear_trade_state()
            return

        # ---- Target filled -> cancel the stop (OCO leg 2) -------------------
        if self.limit_ticket is not None and order_id == self.limit_ticket.order_id:
            if self.stop_ticket is not None and self._is_ticket_working(self.stop_ticket):
                self.stop_ticket.cancel("OCO_TARGET_FILLED")
            self.log(f"EXIT_TARGET | fill={order_event.fill_price:.2f}")
            self._clear_trade_state()
            return

    # =========================================================================
    # 8. HELPERS
    # =========================================================================
    def _halt_day(self, code: str, daily_pnl: float):
        """Flatten everything and lock the algorithm out until 09:30 tomorrow."""
        self.liquidate(tag=code)
        self.transactions.cancel_open_orders(tag=code)
        self._clear_trade_state()
        self.halted_for_day = True
        self.log(f"{code} | daily_pnl={daily_pnl:.2f} "
                 f"equity={self.portfolio.total_portfolio_value:.2f}")

    def _clear_trade_state(self):
        self.entry_ticket = None
        self.stop_ticket = None
        self.limit_ticket = None
        self.entry_price = None
        self.position_direction = 0
        self.breakeven_armed = False

    def _has_working_orders(self) -> bool:
        return len(self.transactions.get_open_orders()) > 0

    @staticmethod
    def _is_ticket_working(ticket) -> bool:
        return ticket.status not in (OrderStatus.Filled,
                                     OrderStatus.Canceled,
                                     OrderStatus.Invalid,
                                     OrderStatus.CancelPending)

    def _round_to_tick(self, price: float) -> float:
        """Snap to the ES 0.25 grid -- off-grid prices are rejected."""
        return round(round(price / self.TICK_SIZE) * self.TICK_SIZE, 2)

    # =========================================================================
    # 9. SHUTDOWN
    # =========================================================================
    def on_end_of_algorithm(self):
        self.log(f"FINAL_EQUITY={self.portfolio.total_portfolio_value:.2f} "
                 f"TOTAL_TRADES={self.transactions.orders_count}")
