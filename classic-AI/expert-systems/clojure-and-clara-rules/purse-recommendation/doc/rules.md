# Rules

This project uses Clara Rules to express recommendation logic declaratively.

A Clara rule has two main parts:

```clojure
(defrule rule-name
  "Optional documentation string."
  conditions
  =>
  actions)
```

The conditions appear before `=>`. The actions appear after `=>`.

## Person facts

The input facts are `Person` records:

```clojure
(defrecord Person [name height favorite-color body-shape])
```

The example inserts two people:

```clojure
(->Person :Emily :tall :pink :rectangle)
(->Person :Sally :petite :blue :rectangle)
```

Each fact has four fields:

- `name`
- `height`
- `favorite-color`
- `body-shape`

## Recommendation facts

The inferred facts are `PurseRecommendation` records:

```clojure
(defrecord PurseRecommendation [purse-size name])
```

A recommendation stores the purse size and the person the recommendation belongs to.

## Tall-person rule

The tall-person rule matches a `Person` whose `height` is `:tall`:

```clojure
(defrule make-a-purse-recommendation-for-rectangular-body-shape-tall-body-height
  "Rule for tall rectangular body shapes."
  [?person <- Person (= height :tall) (= ?name name)]
  =>
  (output-a-recommendation ?person "large" ?name))
```

When the rule matches, it recommends a large purse.

## Petite-person rule

The petite-person rule matches a `Person` whose `height` is `:petite`:

```clojure
(defrule make-a-purse-recommendation-for-rectangular-body-shape-petite-body-height
  "Rule for petite rectangular body shapes."
  [?person <- Person (= height :petite) (= ?name name)]
  =>
  (output-a-recommendation ?person "small" ?name))
```

When the rule matches, it recommends a small purse.

## Cascade rule

The cascade rule demonstrates forward chaining. It matches both a `PurseRecommendation` and the matching `Person`:

```clojure
(defrule show-a-cascade-effect
  "Rule triggered when a purse recommendation is made."
  [?purse-recommendation <- PurseRecommendation (= ?size purse-size) (= ?name name)]
  [?person <- Person (= ?favorite-color favorite-color) (= name ?name)]
  =>
  (output-a-follow-up-recommendation ?name ?favorite-color ?size))
```

This rule can run only after a `PurseRecommendation` fact exists.

The important detail is that `output-a-recommendation` inserts a new fact:

```clojure
(insert! (->PurseRecommendation size name))
```

That new fact gives Clara more information and allows the cascade rule to run.

## Rule conditions

This condition:

```clojure
[?person <- Person (= height :tall) (= ?name name)]
```

means:

- Find a fact of type `Person`.
- Bind the whole fact to `?person`.
- Require the `height` field to equal `:tall`.
- Bind the `name` field to `?name`.

This condition:

```clojure
[?purse-recommendation <- PurseRecommendation (= ?size purse-size) (= ?name name)]
```

means:

- Find a fact of type `PurseRecommendation`.
- Bind the whole recommendation to `?purse-recommendation`.
- Bind the `purse-size` field to `?size`.
- Bind the `name` field to `?name`.

## Rule actions

Rule actions are ordinary Clojure expressions. In this project, the actions call helper functions that print output and sometimes insert additional facts.

The action:

```clojure
(output-a-recommendation ?person "large" ?name)
```

prints a recommendation and inserts a `PurseRecommendation` fact.
