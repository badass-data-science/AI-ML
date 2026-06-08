# Clara Rules in Clojure

Clara Rules is a forward-chaining rules engine for Clojure and ClojureScript. It lets you represent domain knowledge as small, independent rules, insert facts into a working memory, and then fire rules that match those facts.

This README introduces the basic Clara workflow, explains the main concepts, and provides a small runnable example.

## What Is a Rules Engine?

A rules engine is a system for separating conditional decision logic from ordinary application control flow.

Instead of writing one large block of nested `if`, `cond`, or `case` expressions, you define many small rules such as:

- If a customer is VIP, give them priority support.
- If a large purse is recommendated and the customer likes pink, suggest a large pink purse.
- If a support request is high priority, notify the assigned representative.

Each rule describes:

1. The facts it is looking for.
2. The conditions that must be true.
3. The action to perform when the rule matches.

This style is useful when many independent pieces of logic may apply to the same data.

## What Is Clara Rules?

Clara Rules is a Clojure rules engine designed for developers. Rules are written directly in Clojure code rather than in a separate business-rule language.

Clara is especially useful for:

- Expert systems
- Recommendation systems
- Business rules
- Validation logic
- Classification logic
- Alerting and notification logic
- Decision-support systems

A Clara program usually has this flow:

```clojure
create session -> insert facts -> fire rules -> inspect results
```

## Core Concepts

### Facts

Facts are pieces of data inserted into a Clara session.

In Clojure, facts are often represented with records:

```clojure
(defrecord Person [name height favorite-color body-shape])
(defrecord PurseRecommendation [purse-size name])
```

A fact might look like this:

```clojure
(->Person :Emily :tall :pink :rectangle)
```

### Rules

Rules are defined with `defrule`.

A rule has:

- A name
- Optional documentation
- A left-hand side containing conditions
- A right-hand side containing actions

Example:

```clojure
(defrule recommend-large-purse
  "Recommend a large purse for tall people."
  [Person (= height :tall) (= ?name name)]
  =>
  (println ?name "should use a large purse."))
```

The part before `=>` describes what the rule matches.  
The part after `=>` describes what happens when the rule fires.

### Sessions

A Clara session contains:

- The rules to use
- The facts inserted into working memory
- Any derived facts created by rules

A session is created with `mk-session`:

```clojure
(mk-session 'my-project.core)
```

The symbol passed to `mk-session` should name the namespace where your Clara rules are defined.

### Inserting Facts

Facts are inserted with `insert`:

```clojure
(insert session
        (->Person :Emily :tall :pink :rectangle)
        (->Person :Sally :petite :blue :rectangle))
```

### Firing Rules

After inserting facts, call `fire-rules`:

```clojure
(fire-rules session)
```

This causes Clara to evaluate the facts against the rules and run any matching rule actions.

### Derived Facts

Rules can insert new facts. Those new facts may trigger additional rules.

This allows cascading reasoning:

```clojure
(defrule recommend-large-purse
  [Person (= height :tall) (= ?name name)]
  =>
  (insert! (->PurseRecommendation :large ?name)))
```

Then another rule can respond to the new `PurseRecommendation` fact:

```clojure
(defrule print-recommendation
  [PurseRecommendation (= ?size purse-size) (= ?name name)]
  =>
  (println ?name "received a" ?size "purse recommendation."))
```

## Minimal Example

```clojure
(ns clara-example.core
  (:require [clara.rules :refer :all])
  (:gen-class))

(defrecord Person [name height favorite-color body-shape])
(defrecord PurseRecommendation [purse-size name])

(defrule recommend-large-purse
  "Tall people should receive a large purse recommendation."
  [Person (= height :tall) (= ?name name)]
  =>
  (println ?name "should use a large purse.")
  (insert! (->PurseRecommendation :large ?name)))

(defrule recommend-small-purse
  "Petite people should receive a small purse recommendation."
  [Person (= height :petite) (= ?name name)]
  =>
  (println ?name "should use a small purse.")
  (insert! (->PurseRecommendation :small ?name)))

(defrule show-cascade-effect
  "This rule fires after a PurseRecommendation fact has been inserted."
  [PurseRecommendation (= ?size purse-size) (= ?name name)]
  [Person (= ?name name) (= ?color favorite-color)]
  =>
  (println "CASCADE EFFECT:" ?name "should sport a" ?size ?color "purse!"))

(defn -main
  []
  (-> (mk-session 'clara-example.core)
      (insert
       (->Person :Emily :tall :pink :rectangle)
       (->Person :Sally :petite :blue :rectangle))
      (fire-rules)))
```

## Example `project.clj`

```clojure
(defproject clara-example "0.1.0-SNAPSHOT"
  :description "A small Clara Rules example in Clojure"
  :dependencies [[org.clojure/clojure "1.11.1"]
                 [com.cerner/clara-rules "0.21.1"]]
  :main ^:skip-aot clara-example.core
  :target-path "target/%s"
  :profiles {:uberjar {:aot :all}})
```

Depending on your project, you may need to adjust the Clara dependency version.

## Running the Example

From the project root:

```bash
lein run
```

Expected output:

```text
:Emily should use a large purse.
:Sally should use a small purse.
CASCADE EFFECT: :Emily should sport a :large :pink purse!
CASCADE EFFECT: :Sally should sport a :small :blue purse!
```

The exact ordering of rule output may vary depending on the rules and facts.

## Project Layout

A typical Leiningen project using Clara might look like this:

```text
clara-example/
├── project.clj
├── README.md
└── src/
    └── clara_example/
        └── core.clj
```

The namespace in `core.clj` should match the project layout:

```clojure
(ns clara-example.core
  (:require [clara.rules :refer :all])
  (:gen-class))
```

For Clojure namespaces, hyphens in namespace names map to underscores in file paths:

```text
clara-example.core -> src/clara_example/core.clj
```

## When to Use Clara Rules

Clara Rules is a good fit when:

- Many independent rules may apply to the same facts.
- New rules will be added over time.
- Business or expert-system logic should be explicit and readable.
- Rules may trigger additional rules.
- You want to separate decision logic from procedural application code.

## When Not to Use Clara Rules

A rules engine may be unnecessary when:

- The logic is simple and unlikely to grow.
- A few ordinary `if` or `cond` expressions are clearer.
- Rule ordering must be tightly controlled.
- The team is unfamiliar with rules engines and the domain logic is small.

## Glossary

| Term | Meaning |
|---|---|
| Fact | A piece of data inserted into a Clara session |
| Rule | A condition/action pair defined with `defrule` |
| Session | Clara's working environment containing rules and facts |
| Working memory | The collection of facts currently known to the session |
| Forward chaining | Reasoning from known facts toward conclusions |
| Derived fact | A new fact inserted by a rule |
| Query | A way to retrieve information from the session |

## Further Reading

- [Clara Rules official website](https://www.clara-rules.org/)
- [Clara Rules documentation](https://www.clara-rules.org/docs/firststeps/)
- Clara Rules GitHub repository
- Clara examples repository
