(deftemplate person
  (slot name (type SYMBOL))
  (slot height (type SYMBOL))
  (slot favorite-color (type SYMBOL))
  (slot body-shape (type SYMBOL))
)

(deftemplate purse-recommendation
  (slot purse-size (type SYMBOL))
  (slot for-whom (type SYMBOL))
)

(deffacts initial-people
  (person
    (name Emily)
    (height tall)
    (favorite-color pink)
    (body-shape rectangular))

  (person
    (name Sally)
    (height petite)
    (favorite-color blue)
    (body-shape rectangular))
)

(defrule make-a-purse-recommendation-for-rectangular-body-shape-tall-body-height
  (person
    (name ?n)
      (height tall)
      (body-shape rectangular)
    )
  =>
  (printout t ?n " should carry a large purse." crlf)
  (assert (purse-recommendation (purse-size large) (for-whom ?n)))
)

(defrule make-a-purse-recommendation-for-rectangular-body-shape-petite-body-height
  (person
    (name ?n)
    (height petite)
    (body-shape rectangular)
  )
  =>
  (printout t ?n " should carry a small purse." crlf)
  (assert (purse-recommendation (purse-size small) (for-whom ?n)))
)

(defrule show-a-cascade-effect
  (purse-recommendation (purse-size ?q) (for-whom ?n))
  (person (name ?n) (favorite-color ?c))
  =>
  (printout t "CASCADE EFFECT: " ?n " should sport a " ?q ", " ?c " purse!" crlf)
)
