; ============================================================
; The Risk Desk — VIX regime-gated rules
; ============================================================

; REGIME-001a: block risk-on LONG trades during VIX crisis
(defrule block-risk-on-long-in-crisis
  (market-regime (regime crisis) (vix-level ?vix))
  (trade-proposal (pair ?p) (direction long))
  (pair-profile   (pair ?p) (long-character risk-on))
  =>
  (assert (risk-verdict
    (result   BLOCKED)
    (rule-id  "REGIME-001a")
    (reason   (str-cat "VIX crisis (" ?vix "): long " ?p " is a risk-on trade, prohibited during market stress"))
    (severity critical))))

; REGIME-001b: block risk-on SHORT trades during VIX crisis
(defrule block-risk-on-short-in-crisis
  (market-regime (regime crisis) (vix-level ?vix))
  (trade-proposal (pair ?p) (direction short))
  (pair-profile   (pair ?p) (short-character risk-on))
  =>
  (assert (risk-verdict
    (result   BLOCKED)
    (rule-id  "REGIME-001b")
    (reason   (str-cat "VIX crisis (" ?vix "): short " ?p " is a risk-on trade, prohibited during market stress"))
    (severity critical))))

; REGIME-002: reduce position size during elevated VIX
(defrule reduce-size-in-elevated-regime
  (market-regime (regime elevated))
  (trade-proposal (pair ?p) (size-pct ?s&:(> ?s 1.0)))
  =>
  (assert (risk-verdict
    (result   MODIFIED)
    (rule-id  "REGIME-002")
    (reason   (str-cat "Elevated VIX: requested size of " ?s "% exceeds 1.0% limit for elevated regime — reduce position"))
    (severity warning))))
