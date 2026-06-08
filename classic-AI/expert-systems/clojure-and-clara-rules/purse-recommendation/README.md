# Purse Recommendation Expert System

This project is a small Clojure expert system that recommends purse sizes based on facts about people. It uses [Clara Rules](https://www.clara-rules.org/) to define facts, rules, and inferred recommendations.

The example demonstrates a basic rule-based workflow:

1. Create facts about people.
2. Run those facts through a Clara rules session.
3. Match rules against the facts.
4. Print recommendations.
5. Insert a new recommendation fact from one rule.
6. Trigger a follow-up rule from that newly inserted fact.

## Project structure

```text
purse-recommendation/
├── project.clj
├── README.md
├── src/
│   └── purseRecommendation/
│       └── core.clj
├── test/
│   └── purseRecommendation/
│       └── core_test.clj
└── doc/
    ├── intro.md
    ├── architecture.md
    ├── rules.md
    └── extending.md
```

The main application code is in:

```text
src/purseRecommendation/core.clj
```

The Leiningen entry point is configured in `project.clj`:

```clojure
:main ^:skip-aot purseRecommendation.core
```

That namespace must match the namespace declared in `core.clj`:

```clojure
(ns purseRecommendation.core
  (:require [clara.rules :refer :all])
  (:gen-class))
```

## Requirements

Install the following before running the project:

- Java
- Leiningen

You can verify Leiningen is available with:

```bash
lein version
```

## Running the project

From the project root, run:

```bash
lein run
```

Expected output:

```text
:Emily should use a large purse.
:Sally should use a small purse.
CASCADE EFFECT:  :Emily should sport a large :pink purse!
CASCADE EFFECT:  :Sally should sport a small :blue purse!
```

The exact spacing may differ slightly depending on the `println` formatting.

## How the example works

The project defines two record types:

```clojure
(defrecord Person [name height favorite-color body-shape])
(defrecord PurseRecommendation [purse-size name])
```

`Person` facts are inserted into the Clara session:

```clojure
(insert
  (->Person :Emily :tall :pink :rectangle)
  (->Person :Sally :petite :blue :rectangle))
```

Rules inspect those facts. For example, this rule recommends a large purse for a tall person:

```clojure
(defrule make-a-purse-recommendation-for-rectangular-body-shape-tall-body-height
  "Rule for tall rectangular body shapes."
  [?person <- Person (= height :tall) (= ?name name)]
  =>
  (output-a-recommendation ?person "large" ?name))
```

The recommendation function also inserts a new `PurseRecommendation` fact:

```clojure
(insert! (->PurseRecommendation size name))
```

That inserted fact then activates the follow-up rule:

```clojure
(defrule show-a-cascade-effect
  "Rule triggered when a purse recommendation is made."
  [?purse-recommendation <- PurseRecommendation (= ?size purse-size) (= ?name name)]
  [?person <- Person (= ?favorite-color favorite-color) (= name ?name)]
  =>
  (output-a-follow-up-recommendation ?name ?favorite-color ?size))
```

This is the cascade effect: one rule inserts a fact, and another rule reacts to that inserted fact.

## Important namespace note

The namespace used in `mk-session` must be the namespace where the Clara rules are defined:

```clojure
(mk-session 'purseRecommendation.core)
```

Do not use `clara.purseRecommendation` or `clara.purse`. Those namespaces do not exist in this project.

## Building an uberjar

To build a standalone jar:

```bash
lein uberjar
```

Then run the generated standalone jar from the `target/uberjar` directory. The exact filename may vary, but it will look similar to:

```bash
java -jar target/uberjar/purse-0.1.0-SNAPSHOT-standalone.jar
```

## Running tests

Run tests with:

```bash
lein test
```

Note: the generated sample test may still contain placeholder assertions unless it has been updated.

## Documentation

Additional documentation is available in the `doc/` directory:

- [`doc/intro.md`](doc/intro.md) — overview of the project and expert-system idea
- [`doc/architecture.md`](doc/architecture.md) — how the project is organized
- [`doc/rules.md`](doc/rules.md) — explanation of the Clara rules
- [`doc/extending.md`](doc/extending.md) — how to add new facts and rules

## License

Distributed under the Eclipse Public License, either version 1.0 or any later version.
