Submit



Submissions are due by Nov 15, 11:59 PM AoE. You can submit more than once; your latest submission counts.



Submit your project

What you'll need

The email you or a teammate registered with

An abstract of your submission, under 500 words

Your Hugging Face model collection

Add all models, adapters, etc. into a HF collection, with all individual repos and the collection public (un-gated).

Your Hugging Face dataset collection

Add all datasets, logs, traces, benchmarks, curated or synthetic data, etc. into a HF collection, with all individual repos and the collection public.

Your write-up or article

If HF articles are accessible, use them; otherwise, a GitHub repo README is also acceptable.

Your GitHub repo (code or a set of notebooks)

If a repo is used, ensure the README covers reproducibility notes.

If notebooks were used, submit them via a GitHub repo.

The repos and models you used, your track or tracks, and your problem statements

Write-up content

\* required

^ optional

" depends on track

\+ used for hiring decisions, if you opted in / are interested



\[T1] Track 1

\[T2a] Track 2a

\[T2b] Track 2b

\[T3] Track 3



\* Title + one-line claim

\* Tracks + problem statements covered (Bullet list of each track and problem statement covered, with the problem statement number. If "Other", list it as an unnumbered item.)

^ Content warning (If any harmful / NSFW text or examples may appear in the write-up, say so in a highlighted section or subsection at the top.)

\* Abstract (Short description / summary of what the work is about, under 500 words.)

\* Headline before → after number

\* Base model + method in one line

^ "For judges: one minute" box

\* Problem statement and approach

\*+ Why this problem / real use case

^ Research questions or hypotheses

^ Pre-registered hypotheses

^+ Threat model \[T2a] \[T3]

^ Related work (Relevant papers preferred; blogs only if prominent and no equivalent paper exists.)

\* Hackathon repos (kits) used

" CuratorKIT (How it was used and which features.)

" AlignTune (How it was used and which features.)

" AgentTune (How it was used and which features.)

" SafeTune (How it was used and which features.)

" CircuitKIT (How it was used and which features.)

" AuditKIT (How it was used and which features.)

\* How the repos connect (Explanation of how the repos above were used together in the pipeline; optionally attach a chart or diagram.)

\* Models and licenses

\*+ Base Aya model (Which model was used, what config, quant or version, and its license.)

" Teacher model \[T1] (The teacher model or the model distilled from, and the rationale for choosing it.)

" Target / hardened model attacked \[T2a]

^ Closed-model baseline

\* Datasets, benchmarks and licenses

\*+ Source(s) of data (Existing public datasets, a private dataset, a synthetic dataset via distillation or direct generation, etc. If multiple datasets, a table of the distribution by source, task type or language.)

"+ Languages covered (Languages in the training / reference data.)

\*+ Train / test splits, leak check (Train and test sizes, and whether leak checks were done: direct match, embedding based, or MinHash.)

"^+ Safety benchmark(s) \[T2a] \[T2b] (The focus for safety work. Other tracks can optionally use them to show general fine-tuning did not degrade safety. State the languages covered and whether each is part of the focus task or a retention check.)

"^+ General capability benchmark(s) \[T1] \[T3] (The focus for general fine-tuning. Safety tracks can optionally use them to show safety tuning did not drop general performance. State the languages covered and whether each is part of the focus task or a retention check.)

"^+ CuratorKIT filtering \[T1] \[T2] (How CuratorKIT was used and what happened during filtering: what was removed and why, counts before / after.)

"+ Attack prompt set \[T2a] (The adversarial prompts used to attack the model, grouped by attack type such as jailbreak, prompt injection or encoding tricks; their sources, sizes and the languages covered.)

"+ Over-refusal test set \[T2b] (Safe prompts that look risky, used to check the model does not refuse harmless requests; their sources, sizes and the languages covered.)

"+ Tool / task environments, trajectories \[T3]

\* Metrics, and why these

\*+ List of metrics (Each metric used, and whether higher or lower is better.)

"^ Metrics for what should not get worse (Scores showing the change did not break something else, e.g. general benchmark scores after safety tuning, safety scores after general fine-tuning, or the share of harmless prompts wrongly refused.)

^ Checking the automatic judge (If an LLM or classifier scores the outputs: which one, and how often it agrees with human labels on a sample, e.g. % agreement or Cohen's κ.)

" Attack success rate (ASR) and cost-to-break \[T2a] (ASR: share of attacks that got a harmful answer. Cost-to-break: attempts or tokens needed before the first successful attack.)

" Agentic trajectory metrics \[T3] (Task success, tool-call accuracy and average latency, preferably per subset / task / language.)

\* Method / experiments

\*+ Training or attack setup, hyperparameters (Configs and hyperparameters shared, including defaults used for every hackathon repo, with one sub-subsection per repo used, or separate subsections.)

^ Seeds

^ Ablations

\* Results

\* Same model before vs after (table) (Can also compare against the teacher model / closed-source models.)

"^ Safety + general results side by side

^ Confidence intervals (e.g. 95% bootstrap) and significance tests

^ Findings as named subsections

^ What didn't work (negative results)

^ Sample inputs and outputs (Plus summarized logs, e.g. tool calls, where full logs are too long.)

" Results on the quantized / deployed model (GGUF, 4-bit) \[T2b]

^ Cost / ROI

^+ Compute used (Where it came from, which GPUs, how many hours, and the estimated cost to you.)

^+ Cost per 1k replies vs API (Closed-source models compared with: estimate from input/output token counts × cost per M tokens over a small sample, scaled to 1,000. Your own models: tokens/sec (vLLM recommended), time taken for 1k samples × hourly cost at a reasonably priced provider. If the sample count is not 1,000, multiply / divide to compare per thousand.)

\* Limitations

\*+ Known weaknesses (Self-declared gaps, or what hasn't been done so far.)

" Dual-use / ethics \[T2a]

" Responsible disclosure \[T2a]

\* With more credits

\*+ Planned approach (How the team would extend this work with more data / compute / time, or which other experiments could have been done.)

\*+ Expected before / after (Expected / forecast results if the work above is done with additional resources.)

\* Artifacts

\* HF datasets collection (Data, logs, benchmarks; include agentic traces, tool-call logs, and optionally curated and full splits of the data.)

" HF models collection \[T1] \[T2b] \[T3] (Weights and adapters; include GGUFs / quants / mixed-precision quants too, if made.)

\* GitHub repo (If the work was done in notebooks, add them to a repo.)

^ Demo / Space

^ Video / slides

\* Documentation and reproduction

^ Environment requirements (Dependencies and hardware requirements, if any.)

\* Repo layout and run instructions (What each folder / script does and the commands to reproduce the results.)

\* Declaration of work done outside the hackathon (Any part of the submitted work done before or outside the hackathon, if any.)

^ Discussion / what we learned

^ Implications

^ Future work

\* Teammates tagged

^+ Author contributions (Where each author contributed.)

^ Acknowledgements (Any support or grants received, or help / resources used from a person or organisation.)

^ References (Repos and references.)

^ LLM usage statement (Approximate portion of the code / write-up generated by AI.)

^ Changelog (For re-submissions of the same work by the same team before the deadline, list the changes in each version here.)

^ Appendix

^ Prompts, judge rubric

^ Extra tables

" Security considerations \[T2a]

