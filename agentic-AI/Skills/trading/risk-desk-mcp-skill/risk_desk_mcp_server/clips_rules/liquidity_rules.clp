; ============================================================
; The Risk Desk — liquidity rules
; ============================================================

; LIQ-001: block trades with excessive spread (any pair, any session)
(defrule block-excessive-spread
  (pair-liquidity (pair ?p) (spread-pips ?s&:(> ?s 3.0)))
  (trade-proposal (pair ?p))
  =>
  (assert (risk-verdict
    (result   BLOCKED)
    (rule-id  "LIQ-001")
    (reason   (str-cat "Liquidity: " ?p " spread of " ?s " pips exceeds 3.0 pip threshold — market too thin to enter"))
    (severity critical))))

; LIQ-002: warn when trading European majors during Tokyo session
; AUD/USD and NZD/USD are excluded — they have natural liquidity during Asian hours
(defrule warn-european-pairs-in-tokyo
  (pair-liquidity (pair ?p&:(or (eq ?p "EUR/USD") (eq ?p "GBP/USD"))) (session tokyo))
  (trade-proposal (pair ?p))
  =>
  (assert (risk-verdict
    (result   APPROVED)
    (rule-id  "LIQ-002")
    (reason   (str-cat ?p " during Tokyo session: European pair liquidity is typically thin — verify spread before entry"))
    (severity info))))
