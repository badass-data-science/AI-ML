(ns purseRecommendation.core
  (:require [clara.rules :refer :all])
  (:gen-class))

;; Person
(defrecord Person [name height favorite-color body-shape])

;; Purse Recommendation
(defrecord PurseRecommendation [purse-size name])

(defn output-a-recommendation
      "Outputs the initial purse recommendation."
      [person size name]
      (println name "should use a" size "purse.")
      (insert! (->PurseRecommendation size name))
)

(defn output-a-follow-up-recommendation
      "This demonstrates how when a fact is inserted by a function, it automatically triggers rules related to the fact."
      [name color size]
      (println "CASCADE EFFECT: " name "should sport a" size color "purse!")
)

(defrule make-a-purse-recommendation-for-rectangular-body-shape-tall-body-height
	  "Rule for tall rectangular body shapes."
         [?person <- Person (= height :tall) (= ?name name)]
=>
        (output-a-recommendation ?person "large" ?name)
)

(defrule make-a-purse-recommendation-for-rectangular-body-shape-petite-body-height
         "Rule for petite rectangular body shapes."
         [?person <- Person (= height :petite) (= ?name name)]
=>
        (output-a-recommendation ?person "small" ?name)
)

(defrule show-a-cascade-effect
	  "Rule triggered when a	purse recommendation is	made."
        [?purse-recommendation <- PurseRecommendation (= ?size purse-size) (= ?name name)]
        [?person <- Person (= ?favorite-color favorite-color) (= name ?name)]
=>
        (output-a-follow-up-recommendation ?name ?favorite-color ?size)
)

(defn -main
  "Run this example."
  []
  (-> (mk-session 'purseRecommendation.core)
      (insert
        (->Person :Emily :tall :pink :rectangle)
        (->Person :Sally :petite :blue :rectangle))
      (fire-rules)))
