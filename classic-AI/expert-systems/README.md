This directory contains my experiments with expert systems:

# What Is an Expert System?

An expert system is a computer program that uses rules, facts, and logical reasoning to solve problems or make recommendations in a specific subject area.

Expert systems are one of the classic forms of artificial intelligence. Instead of learning from large amounts of data, an expert system usually relies on knowledge written by humans. This knowledge is often represented as facts and rules.

## Basic Idea

An expert system tries to imitate the decision-making process of a human expert.

For example, a human fashion consultant might say:

> If a person has a rectangular body shape and is tall, then they should carry a large purse.

An expert system can represent that same idea as a rule:

```clips
IF body-shape is rectangular
AND height is tall
THEN recommend a large purse
```

The system can then apply this rule automatically to people whose facts match the rule conditions.

## Main Parts of an Expert System

Most expert systems have three major parts:

1. A knowledge base
2. A working memory
3. An inference engine

## Knowledge Base

The knowledge base stores the system’s expert knowledge.

This knowledge usually includes rules such as:

```text
IF a person is tall
THEN recommend a large purse
```

or:

```text
IF a patient has a fever and a cough
THEN consider the possibility of an infection
```

The knowledge base is the part of the system that contains the logic of the expert domain.

## Working Memory

Working memory stores the facts that are currently known.

For example, in a purse recommendation system, working memory might contain facts like:

```text
Emily is tall.
Emily’s favorite color is pink.
Emily has a rectangular body shape.
```

The inference engine compares these facts against the rules in the knowledge base.

## Inference Engine

The inference engine is the reasoning part of the expert system.

It looks at the facts in working memory, checks which rules match those facts, and then fires the matching rules.

When a rule fires, it may:

* print a recommendation
* assert a new fact
* modify an existing fact
* trigger more rules

This is how expert systems can create a chain of reasoning.

## Rules

Rules are usually written in an `IF` / `THEN` form.

For example:

```text
IF a person is tall
AND the person has a rectangular body shape
THEN recommend a large purse
```

The `IF` part contains the conditions.

The `THEN` part contains the action.

In CLIPS, a rule might look like this:

```clips
(defrule recommend-large-purse
  (person
    (name ?n)
    (height tall)
    (body-shape rectangular))
  =>
  (printout t ?n " should carry a large purse." crlf)
)
```

This rule says:

* Find a `person`.
* Store the person’s name in the variable `?n`.
* Check that the person’s height is `tall`.
* Check that the person’s body shape is `rectangular`.
* If all conditions match, print a recommendation.

## Facts

Facts describe what the system currently knows.

For example:

```clips
(person
  (name Emily)
  (height tall)
  (favorite-color pink)
  (body-shape rectangular))
```

This fact tells the system that there is a person named Emily who is tall, likes pink, and has a rectangular body shape.

Facts are matched against rules.

## Forward Chaining

Many expert systems use a reasoning method called forward chaining.

Forward chaining starts with known facts and uses rules to infer new facts or conclusions.

For example:

```text
Known fact:
Emily is tall and has a rectangular body shape.

Rule:
If someone is tall and has a rectangular body shape, recommend a large purse.

Conclusion:
Emily should carry a large purse.
```

The new conclusion can also become a new fact, which may trigger additional rules.

This creates a cascade effect.

## Cascade Effects

A cascade effect happens when one rule creates a new fact that causes another rule to fire.

For example:

```text
Rule 1:
If Emily is tall and rectangular, recommend a large purse.

New fact:
Emily has a large purse recommendation.

Rule 2:
If someone has a purse recommendation and a favorite color, recommend a purse in that color.

Final recommendation:
Emily should sport a large, pink purse.
```

This is one of the powerful features of expert systems. Simple rules can combine to produce more detailed conclusions.

## Why Use Expert Systems?

Expert systems are useful when a problem can be described clearly with rules.

They are often used for:

* diagnosis
* classification
* recommendation
* troubleshooting
* planning
* decision support

Examples include:

* medical diagnosis systems
* equipment troubleshooting systems
* financial decision systems
* configuration systems
* eligibility screening systems
* recommendation systems

## Strengths of Expert Systems

Expert systems have several advantages:

* Their reasoning can be easier to inspect than some machine learning models.
* Rules can be added, removed, or edited directly.
* They work well for domains with clear decision logic.
* They can explain decisions by showing which rules fired.
* They do not require large training datasets.

## Limitations of Expert Systems

Expert systems also have limitations:

* They depend on the quality of the rules.
* They can become difficult to maintain as the number of rules grows.
* They usually do not learn automatically from data.
* They may not handle ambiguous or incomplete information well unless explicitly designed to do so.
* They can be brittle when facing situations not covered by the rules.

## Expert Systems vs. Machine Learning

Expert systems and machine learning systems solve problems in different ways.

An expert system uses rules written by people:

```text
IF condition
THEN action
```

A machine learning system usually learns patterns from data.

For example, an expert system might say:

```text
IF height is tall
THEN recommend a large purse
```

A machine learning system might look at thousands of previous fashion recommendations and learn statistical patterns from them.

Expert systems are often better when the rules are clear and explainability is important. Machine learning is often better when the rules are hard to write manually but patterns can be learned from data.

## CLIPS and Expert Systems

CLIPS is a programming language and environment for building expert systems.

CLIPS provides tools for:

* defining fact templates
* asserting facts
* writing rules
* running an inference engine
* inspecting facts and rule activations

A small CLIPS expert system may contain:

```clips
(deftemplate person
  (slot name)
  (slot height)
  (slot favorite-color)
  (slot body-shape))
```

This defines the structure of a `person` fact.

Rules can then match against these facts and produce conclusions.

## Example Expert System Workflow

A typical expert system workflow looks like this:

1. Define templates for structured facts.
2. Add initial facts.
3. Define rules.
4. Reset the system to load initial facts into working memory.
5. Run the inference engine.
6. Review the recommendations or conclusions.

In CLIPS, the final commands often look like:

```clips
(load "example.clp")
(reset)
(run)
```

## Summary

An expert system is a rule-based artificial intelligence program that reasons from facts to conclusions.

It usually contains:

* facts describing the current situation
* rules describing expert knowledge
* an inference engine that applies the rules

Expert systems are useful when knowledge can be written clearly as logical rules. They are especially helpful for recommendation, diagnosis, classification, and troubleshooting tasks where explainable reasoning is important.
