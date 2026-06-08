# Introduction

This project is a small expert system written in Clojure. It uses Clara Rules to recommend purse sizes based on facts about people.

An expert system is a program that represents decision-making knowledge as facts and rules. Instead of writing one long sequence of `if` statements, the program stores information as facts and lets a rule engine decide which rules apply.

In this project, the facts are people and purse recommendations. The rules look at each person's characteristics and infer a purse-size recommendation.

## What the example demonstrates

The project demonstrates four core expert-system ideas:

1. **Facts** — structured pieces of information, such as a person named Emily who is tall and likes pink.
2. **Rules** — decision logic that matches facts, such as recommending a large purse for a tall person.
3. **Inference** — the rule engine derives a new fact or conclusion from existing facts.
4. **Chaining** — one rule can insert a new fact that causes another rule to run.

The chaining behavior is especially important. The first set of rules creates a `PurseRecommendation`. The follow-up rule then notices that recommendation and prints an additional message using the person's favorite color.

## Main files

The most important files are:

```text
project.clj
src/purseRecommendation/core.clj
```

`project.clj` configures Leiningen and declares Clara Rules as a dependency.

`core.clj` defines the facts, rules, helper functions, and the `-main` function that runs the example.

## Running the example

From the project root, run:

```bash
lein run
```

This creates a Clara session, inserts example `Person` facts, and fires all matching rules.
