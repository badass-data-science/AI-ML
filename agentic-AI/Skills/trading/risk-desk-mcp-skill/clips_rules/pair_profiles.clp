; ============================================================
; The Risk Desk — static risk profiles for the seven major pairs
;
; long-character:  risk character when going long (buying base)
; short-character: risk character when going short (selling base)
;
; USD/CAD is the notable inversion: long USD/CAD is risk-off
; because you are buying the safe-haven USD against a commodity
; currency.  USD/CHF is neutral-long because both currencies
; have safe-haven characteristics; the short (buying CHF) is
; the cleaner risk-off signal.
; ============================================================

(deffacts major-pair-profiles
  (pair-profile (pair "EUR/USD") (long-character risk-on)  (short-character risk-off))
  (pair-profile (pair "GBP/USD") (long-character risk-on)  (short-character risk-off))
  (pair-profile (pair "AUD/USD") (long-character risk-on)  (short-character risk-off))
  (pair-profile (pair "NZD/USD") (long-character risk-on)  (short-character risk-off))
  (pair-profile (pair "USD/JPY") (long-character risk-on)  (short-character risk-off))
  (pair-profile (pair "USD/CHF") (long-character neutral)  (short-character risk-off))
  (pair-profile (pair "USD/CAD") (long-character risk-off) (short-character risk-on)))
