# Assignment 3 Report

- **Name**:
- **Student ID**:
- **Group**: [BBADBA 5A | BBADBA 5B | PPLEDBA 5A]

## Use case

*Who asks, what they do with the answer, and what a wrong answer costs compared with "I don't know".*

## Corpus

*Table: document, URL, date, licence, words.*

## Ingestion

*How you extract, clean and chunk; one extraction problem you fixed; chunk metadata; number of chunks.*

## Test set

*How you wrote it, counts by kind, two examples.*

## Retrieval

*Hit rate at 1/3/5 per design, words added to the prompt, the design you chose.*

## Tools and subagent

 ▐▛███▛█   Claude Code v2.1.322
▝▜██████▀  Opus 5.5 · Claude Enterprise
 ▝▝   ▝▝   ~/Assignment 3 - AI/daiist-assignment-3


❯ /agents                                                                       
  ⎿  The /agents wizard has been removed.

     Ask Claude to create or update subagents for you (e.g. "create a code-reviewer subagent that ..."),
     or edit the files directly:
       • .claude/agents/       (this project)
       • ~/.claude/agents/     (all projects)

     Docs: https://code.claude.com/docs/en/sub-agents

❯ Use the corpus-researcher subagent: does an employer have to tell workers     
  before using an AI system to monitor them?                                    

⏺ corpus-researcher(Employer AI monitoring notice)
  ⎿  Backgrounded agent (↓ to manage · ctrl+o to expand)

⏺ The corpus-researcher subagent is now searching the AI Act and the GDPR
  separately for this question. I'll send you its answer, with citations, when
  it finishes.

✻ Brewed for 7s · done 4:31 PM

› Message from corpus-researcher
  ANSWER

  Yes. Under the AI Act, an employer that uses a high-risk AI system at the
  workplace must inform workers' representatives and the affected workers before
  putting the system into service or using it [ai_act, Article 26]. AI systems
  used to monitor and evaluate the performance and behaviour of workers count as
  high-risk [ai_act, Annex III]. Under the GDPR, workers must also receive
  information about the processing of their personal data [gdpr, Article 13]
  [gdpr, Article 14].

  REASONING (brief)

  1. Is workplace monitoring high-risk? Yes. Annex III, point 4(b) lists AI
  systems used "to monitor and evaluate the performance and behaviour of persons
  in such [work-related] relationships" [ai_act, Annex III].

  2. Who must be told, and when? Article 26(7) says that "before putting into
  service or using a high-risk AI system at the workplace," employers must
  inform workers' representatives and the affected workers. This follows Union
  and national rules and practice on informing workers [ai_act, Article 26].

  3. Individual notice when the system supports decisions. Deployers of Annex
  III high-risk systems that make, or help make, decisions about people must
  tell those people that the system is being used [ai_act, Article 26].

  4. Emotion recognition is banned, not just subject to notice. Using AI to
  infer the emotions of a person in the workplace is prohibited, except for
  medical or safety reasons [ai_act, Article 5]. Outside that ban, anyone using
  an emotion recognition or biometric categorisation system must tell the people
  exposed to it that it is operating [ai_act, Article 50].

  5. GDPR. When personal data are collected from workers, the employer must
  provide the required information at the time of collection. This includes the
  purposes, the legal basis and, where it applies, the existence of automated
  decision-making, with meaningful information about the logic involved and the
  likely consequences [gdpr, Article 13]. If the data come from somewhere else,
  the employer must provide similar information, at the latest within one month
  [gdpr, Article 14]. Processing must be transparent [gdpr, Article 5]. Member
  States may set more specific rules for employee data. These rules must include
  safeguards on transparency and on monitoring systems at the workplace [gdpr,
  Article 88].

  LIMITS

  - Article 26(7) covers high-risk systems only. For a monitoring tool that is
  not high-risk, the passages show only the GDPR duties to inform, plus the
  Article 50 duty if emotion recognition or biometric categorisation is
  involved.
  - The corpus does not include national (e.g. Spanish) rules on informing
  workers' representatives, which Article 26(7) and GDPR Article 88 refer to. It
  also has no recitals and no AI Act application dates, so it cannot say from
  what date Article 26 applies.
  - I did not retrieve the text of GDPR Article 22. It is mentioned only through
  the cross-reference in Articles 13(2)(f) and 14(2)(g).

  CITATIONS RETRIEVED

  - [ai_act, Article 26] (paras 7 and 11 are relevant; paras 1-6, 8-10 and 12
  also came up)
  - [ai_act, Annex III] (point 4(b) employment monitoring; point 1(c) emotion
  recognition)
  - [ai_act, Article 5] (para 1(f) workplace emotion recognition ban)
  - [ai_act, Article 50] (para 3 notice for emotion recognition and biometric
  categorisation)
  - [ai_act, Article 3] (definition 39, emotion recognition system)
  - [ai_act, Article 13] (provider transparency to deployers; not central)
  - [gdpr, Article 13]
  - [gdpr, Article 14]
  - [gdpr, Article 5]
  - [gdpr, Article 88]

⏺ Yes. Under the AI Act, an employer must inform workers' representatives and
  the affected workers before it puts a high-risk AI system into service or uses
  it at the workplace [ai_act, Article 26]. AI systems that monitor and
  evaluate the performance and behaviour of workers are classed as high-risk
  [ai_act, Annex III]. The GDPR also requires the employer to give workers
  information about how their personal data are processed [gdpr, Article 13]
  [gdpr, Article 14].

  Why:
  - Monitoring is high-risk. Annex III, point 4(b) covers AI used "to monitor
    and evaluate the performance and behaviour of persons" in work relationships
    [ai_act, Annex III].
  - Notice comes before use. Article 26(7) requires the notice before the system
    is put into service or used, following Union and national rules on
    informing workers [ai_act, Article 26].
  - People affected by decisions must be told. If the system makes or helps make
    decisions about a person, the employer must also tell that person the
    system is being used [ai_act, Article 26].
  - GDPR information duties. When data are collected from the worker, the
    employer must give the information at that moment. It includes the purposes,
    the legal basis and, where it applies, automated decision-making and the
    logic behind it [gdpr, Article 13]. If the data come from elsewhere, the
    information must be given within one month at the latest [gdpr, Article 14].
    Processing must be transparent [gdpr, Article 5]. Member States may set
    more specific rules for employee data, with safeguards on transparency and
    on workplace monitoring systems [gdpr, Article 88].
  - Emotion recognition is banned at work. Using AI to infer a worker's emotions
    is prohibited, except for medical or safety reasons. Telling workers does
    not make it allowed [ai_act, Article 5]. Where emotion recognition or
    biometric categorisation is allowed, the people exposed to it must be told
    it is operating [ai_act, Article 50].

  Limits:
  - Article 26(7) applies only to high-risk systems. For a monitoring tool that
    is not high-risk, the documents show only the GDPR duties, plus Article 50
    if emotion recognition or biometric categorisation is involved.
  - The documents do not include Spanish or other national rules on informing
    workers' representatives, even though Article 26 and GDPR Article 88 refer
    to them. They also have no recitals and no AI Act application dates.
  - The text of GDPR Article 22 was not retrieved. It appears only as a
    cross-reference inside Articles 13 and 14.

  HR should check the cited articles before making any decision. I can give you
  a more detailed explanation of any of these points.

✻ Crunched for 8s · done 4:31 PM

⏺ Agent "Employer AI monitoring notice" finished · 23s

## Adding a document

*What happens when a document is already indexed. The unanswerable question, the document you added, the answer before and after.*

## Harness and guardrails

*Your AGENTS.md rules, the approval step, and your extra guardrail with one case where it fired.*

## Evaluation

*Code checks, the judge's rubric, your agreement with the judge, and results: correct, unsupported, correct refusals, wrong refusals.*

## Failures

*Failed questions by cause, with counts, and what to fix first.*

## Decision

*Would you let users rely on it, for what, with which safeguard? Cost and latency per question.*

## AI use

*What you used and how, or "none".*
