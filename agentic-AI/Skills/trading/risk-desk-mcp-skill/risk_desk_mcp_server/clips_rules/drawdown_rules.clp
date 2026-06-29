; ============================================================
; The Risk Desk — drawdown-based circuit breaker rules
; ============================================================

; DD-001: hard circuit breaker — all new positions halted
(defrule drawdown-circuit-breaker-hard
  (account-state (weekly-drawdown-pct ?d&:(>= ?d 5.0)))
  (trade-proposal)
  =>
  (assert (risk-verdict
    (result   BLOCKED)
    (rule-id  "DD-001")
    (reason   (str-cat "Circuit breaker: weekly drawdown of " ?d "% meets or exceeds the 5% hard limit — no new positions"))
    (severity critical))))

; DD-002: soft warning — approaching the circuit breaker
(defrule drawdown-soft-warning
  (account-state (weekly-drawdown-pct ?d&:(and (>= ?d 3.0) (< ?d 5.0))))
  (trade-proposal)
  =>
  (assert (risk-verdict
    (result   APPROVED)
    (rule-id  "DD-002")
    (reason   (str-cat "Warning: weekly drawdown at " ?d "% — approaching 5% circuit breaker, reduce size accordingly"))
    (severity warning))))
