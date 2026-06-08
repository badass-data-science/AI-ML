# Architecture

The project has a simple Leiningen/Clojure layout:

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

## `project.clj`

`project.clj` declares the project name, dependencies, main namespace, and build settings.

The key dependency is Clara Rules:

```clojure
[com.cerner/clara-rules "0.17.0"]
```

The important entry-point setting is:

```clojure
:main ^:skip-aot purseRecommendation.core
```

That tells Leiningen to run the `-main` function in the `purseRecommendation.core` namespace.

## `src/purseRecommendation/core.clj`

This file contains the full expert system:

- The namespace declaration
- Record definitions for facts
- Helper functions that print results and insert inferred facts
- Clara `defrule` definitions
- The `-main` function

The namespace declaration is:

```clojure
(ns purseRecommendation.core
  (:require [clara.rules :refer :all])
  (:gen-class))
```

The `:require` imports Clara's rule macros and functions, including `defrule`, `mk-session`, `insert`, `insert!`, and `fire-rules`.

## Fact model

The project uses Clojure records as Clara fact types.

```clojure
(defrecord Person [name height favorite-color body-shape])
(defrecord PurseRecommendation [purse-size name])
```

A `Person` fact represents input data.

A `PurseRecommendation` fact represents an inferred result.

## Runtime flow

The runtime flow is:

1. `lein run` calls `purseRecommendation.core/-main`.
2. `-main` creates a Clara session with `(mk-session 'purseRecommendation.core)`.
3. `-main` inserts two `Person` facts.
4. `fire-rules` evaluates all rules.
5. Matching rules print recommendations.
6. Recommendation rules insert `PurseRecommendation` facts.
7. The follow-up rule reacts to those inserted facts.

## Namespace alignment

Several names must remain aligned:

```text
project.clj :main       purseRecommendation.core
core.clj namespace      purseRecommendation.core
mk-session namespace    purseRecommendation.core
```

If one of these names does not match, Leiningen or Clara may not be able to find the code or rules.
