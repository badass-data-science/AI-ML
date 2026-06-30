; ============================================================
; The Risk Desk — open-position concentration rules
; ============================================================

; CONC-001: hard block when too many positions are already open
(defrule concentration-hard-block
  (account-state (open-positions ?n&:(>= ?n 5)))
  (trade-proposal)
  =>
  (assert (risk-verdict
    (result   BLOCKED)
    (rule-id  "CONC-001")
    (reason   (str-cat "Concentration: " ?n " positions already open — hard limit of 5 reached, no new entries"))
    (severity critical))))

; CONC-002: soft warning when approaching the concentration limit
(defrule concentration-soft-warning
  (account-state (open-positions ?n&:(and (>= ?n 3) (< ?n 5))))
  (trade-proposal)
  =>
  (assert (risk-verdict
    (result   APPROVED)
    (rule-id  "CONC-002")
    (reason   (str-cat "Concentration: " ?n " positions open — approaching 5-position limit, reduce size accordingly"))
    (severity warning))))
