# Extending the Expert System

This project is intentionally small so it is easy to modify. You can extend it by adding new facts, new fields, new people, or new rules.

## Add another person

To add another person to the example, edit the `insert` call inside `-main`:

```clojure
(insert
  (->Person :Emily :tall :pink :rectangle)
  (->Person :Sally :petite :blue :rectangle)
  (->Person :Jordan :tall :green :rectangle))
```

Then run:

```bash
lein run
```

If the new person matches an existing rule, Clara will print a recommendation.

## Add a new height category

Suppose you want a medium purse recommendation for people with average height. Add a new person:

```clojure
(->Person :Alex :average :purple :rectangle)
```

Then add a new rule:

```clojure
(defrule make-a-purse-recommendation-for-average-body-height
  "Rule for average-height people."
  [?person <- Person (= height :average) (= ?name name)]
  =>
  (output-a-recommendation ?person "medium" ?name))
```

Now Clara can infer a medium purse recommendation for that person.

## Add body-shape-specific logic

The current rule names mention rectangular body shape, but the rule conditions only check `height`.

To make a rule require both height and body shape, include `body-shape` in the condition:

```clojure
(defrule make-a-purse-recommendation-for-tall-rectangular-body-shape
  "Rule for tall people with rectangular body shape."
  [?person <- Person
   (= height :tall)
   (= body-shape :rectangle)
   (= ?name name)]
  =>
  (output-a-recommendation ?person "large" ?name))
```

This rule fires only for a person whose height is `:tall` and whose body shape is `:rectangle`.

## Add a new fact type

You can add more fact types with `defrecord`. For example, you might define style preferences:

```clojure
(defrecord StylePreference [name style])
```

Then insert a style preference:

```clojure
(->StylePreference :Emily :formal)
```

A rule could then match both a person and a style preference:

```clojure
(defrule formal-style-follow-up
  [?person <- Person (= ?name name)]
  [?style <- StylePreference (= name ?name) (= style :formal)]
  =>
  (println ?name "may prefer a more structured purse design."))
```

## Keep namespaces consistent

If you rename the project or namespace, update all of these places together:

```text
project.clj :main
core.clj ns declaration
mk-session namespace
any test namespace requires
```

For the current project, the correct namespace is:

```clojure
purseRecommendation.core
```

The session should be created with:

```clojure
(mk-session 'purseRecommendation.core)
```

## Troubleshooting

### `Cannot find anything to run`

This usually means `project.clj` points to a namespace that does not exist or does not contain `-main`.

Check this line in `project.clj`:

```clojure
:main ^:skip-aot purseRecommendation.core
```

Then check that `core.clj` contains:

```clojure
(ns purseRecommendation.core
  ...)

(defn -main
  ...)
```

### `No namespace: clara.purseRecommendation found`

This means `mk-session` was pointed at a namespace that does not exist.

Use:

```clojure
(mk-session 'purseRecommendation.core)
```

Do not use:

```clojure
(mk-session 'clara.purseRecommendation)
```

`clara.rules` is the library namespace. `purseRecommendation.core` is the application namespace containing this project's rules.
