; ============================================================
; The Risk Desk — CLIPS fact templates
; ============================================================

; Trade proposed by the LLM agent
(deftemplate trade-proposal
  (slot pair      (type STRING))
  (slot direction (type SYMBOL) (allowed-symbols long short))
  (slot size-pct  (type FLOAT)))

; Market regime supplied by the VIX fuzzy MCP skill
(deftemplate market-regime
  (slot vix-level (type FLOAT))
  (slot regime    (type SYMBOL) (allowed-symbols calm normal elevated crisis)))

; Account state supplied by the broker MCP skill
(deftemplate account-state
  (slot balance              (type FLOAT))
  (slot weekly-drawdown-pct  (type FLOAT))
  (slot open-positions       (type INTEGER)))

; Pair liquidity supplied by the forex data source
(deftemplate pair-liquidity
  (slot pair        (type STRING))
  (slot spread-pips (type FLOAT))
  (slot session     (type SYMBOL) (allowed-symbols london new-york tokyo sydney overlap)))

; Static risk character of each pair by direction
(deftemplate pair-profile
  (slot pair            (type STRING))
  (slot long-character  (type SYMBOL) (allowed-symbols risk-on risk-off neutral))
  (slot short-character (type SYMBOL) (allowed-symbols risk-on risk-off neutral)))

; Verdict asserted by a fired rule
(deftemplate risk-verdict
  (slot result   (type SYMBOL) (allowed-symbols APPROVED BLOCKED MODIFIED))
  (slot rule-id  (type STRING))
  (slot reason   (type STRING))
  (slot severity (type SYMBOL) (allowed-symbols info warning critical)))
