---
name: modeling-with-dynestyx
description: Grills the user about a system they describe, then builds a dynestyx probabilistic dynamical model from it. Delivers a folder with a LaTeX PDF that states the problem setting, the exact mathematical definition of the model, the interpretation of each component, and a minimal numpyro-style dynestyx model function. Use when the user wants to model a dynamical or state-space system in dynestyx, points to a paper or codebase describing such a system, or asks to be grilled on a dynestyx model, even if they do not name this skill.
---

# Workflow

The user describes a system, or points to a paper or codebase that describes one, that can be cast as a probabilistic dynamical model. The deliverable folder holds the compiled PDF and a minimal model function at its top level. Keep the LaTeX source and every file the compilation produces in a subfolder of the deliverable, so the build clutter stays out of the way. The PDF is the shared document that you and the user improve together until the model is correct. Work in three stages.

Sourcing the real data the model runs on is a hard requirement that the build depends on. Read Source the real data below before you reach the build.

First read the dynestyx documentation so you know which modeling choices the library forces a user to make. Start with docs/index.md and docs/tutorials.md, then docs/api_reference/.

## Stage 1. Draft the PDF

From the user's description or the source they point to, write your best first attempt at the model and compile it to a PDF right away, before resolving every detail and before writing any model code. Do not wait for a complete picture. Where information is missing or you had to assume something, write the gap into the PDF as a marked open question.

Put every equation in the PDF, never in the chat. LaTeX does not render in the chat, and the math is the part the user most needs to read. Lay the PDF out as described under Document structure below.

## Stage 2. Grill and refine the PDF

Interview the user to close the open questions and correct the model. Run the interview as ordinary chat and anchor every question to a place in the PDF. Do not use any structured question or multiple choice tool, and do not restate equations in the chat. Point the user at the relevant part of the PDF instead.

Resolve each of these.

- The problem setting and the question the model answers.
- Every variable in the model, captured in the variables table that Document structure below specifies.
- The initial distribution of the latent state.
- The state evolution. Settle whether time is discrete or continuous, whether the dynamics are deterministic or carry process noise, and whether external controls enter.
- The observation model. Settle which variables are observed, the form of the observation noise, and whether observations arrive at irregular times.
- How to verify the model. The shape check in Stage 3 is not enough on its own. Settle on a real verification and grill the user on which of two routes fits. A prior predictive check draws from the model, reduces the draws to summaries, and asks whether each data summary sits within the spread of the simulated ones. Choose summaries the data can speak to, such as the spread of one-step displacements or the count of observations per unit time. A figure recreation reproduces a specific figure from the paper, such as a distribution it plots or a trajectory it shows. Settle the route, the dataset or figure it targets, and what counts as agreement.

After each round, revise the LaTeX, recompile, and show the PDF again. Repeat until the user confirms it is correct. A wrong PDF propagates into wrong code, so do not begin Stage 3 until the user approves the PDF.

Only the user can declare the grilling finished. After each round, ask directly whether each open question is settled and treat it as closed only when the user says so. When you believe the model is complete, say so and ask the user to confirm, then hold until the user gives that confirmation in plain words. Silence or the absence of objection does not count as approval.

## Source the real data

The model and its verification run on real data, and sourcing that data is a hard requirement. Before you build the model you must find and obtain the actual datasets, maps, coordinates, and other fixed inputs it needs. A synthetic or stand-in substitute is a last resort. Use one only after you have genuinely tried and failed to obtain the real data, and when you do, tell the user plainly that the asset is a substitute.

Never assume that data is unavailable. Read the paper's data availability statement and follow every archive it names, such as a Zenodo or Dryad deposit or a code repository. A code archive often ships code alone, while the data sits in a separate archive under its own identifier, so a missing file in one place does not mean the data is gone. Where the paper builds an asset from a public source, such as a government GIS portal or an open data service, rebuild it from that source. Keep the data the model is fit to, the observations, separate from the complementary datasets that only set parameter values, and source each from its own origin. When the first route fails, try the cited source studies, the corresponding author's repositories, and public portals before you settle for a substitute.

## Stage 3. Build and verify the model

After the user approves the PDF, add the model to the folder. It is a minimal numpyro-style python function that holds only the dynestyx model definition and follows the model the PDF describes. Build it on the real data you sourced, not on placeholders. Match the dynestyx API you read in the docs.

Expose the model through a single builder that returns a dynestyx model object, so the definition lives in one place. Every piece of downstream code builds the model through that builder and drives it with the dynestyx simulators and inference functions. The verification, any figure the model reproduces, and any later inference all take this path. Downstream code does not rebuild the model's internal distribution classes by hand, since that bypasses the library and duplicates the definition. Give the state evolution and the observation model their proper dynestyx base classes so the type-based machinery, such as the auto-selecting simulator, routes the model on its own.

Do not stop at writing the function. Verify that it runs and generates reasonable data before you hand it over. Verification has two parts. First, draw a forward simulation and check that the sampled states and observations carry the shapes the PDF implies. Second, run the verification the PDF specifies, for a prior predictive check by comparing each data summary against the spread of many simulated trajectories, for a figure recreation by generating the figure's quantity from the model and placing it beside the paper's version. Report the result and flag any disagreement. Keep both parts in the folder as a small script or test so the user can rerun them, and run everything through the project uv environment. When the model fails to build or sample, fix it and verify again. When the verification disagrees with the paper or the data, surface it to the user and settle it before you report the model as done.

## Document structure

The PDF follows a fixed section order.

1. Problem setting. A crisp statement of the system, the question the model answers, and the time index. Keep it short.
2. Variables. A complete account of every quantity in the model, laid out as described below.
3. The model. The initial distribution of the latent state, the state evolution, and the observation model, each written as a labelled equation.
4. Verification. The verification the model will pass and its target. For a prior predictive check, the summaries compared against the available data, each with the value the data takes and its source. For a figure recreation, the paper figure to reproduce, its locator, and the quantity it shows.

Begin the variables section by writing the latent state vector as a display equation, so the state space the model evolves is defined inside this section rather than left to prose elsewhere. Then give one table that accounts for every variable, with these columns.

- Symbol. The notation used in the equations.
- Meaning. A short interpretation in the problem.
- Dimension and units. The size of the variable and its physical units.
- Role. One of latent state, latent increment, observed data, fixed input, or hyperparameter.
- Value. The concrete value for every fixed scalar, and a dash for anything that has no single value.
- Source. The exact location in the paper or supplement where the variable is defined or its value is fixed, given as an equation, section, table, or figure number, with a page when that helps the reader find it.

A latent variable points to the equation that defines it. A hyperparameter or other constant points to the equation, table, or passage that fixes its value, and where the paper attributes that value to a complementary dataset or to earlier work, name that origin next to the in-paper locator and carry the original citation through. Observed data point to the passage that describes the dataset, with the archive identifier or repository link that the paper provides. Do not put a code filename or an internal artifact in this column when the paper or supplement states the same fact. When an entry runs longer than a short phrase, keep the locator in the cell and add a short sourcing subsection after the table for the detail.

Order the rows by role, with the latent state components first and the hyperparameters last.

The verification section records the route chosen with the user. For a prior predictive check, give one table that lists each summary, with these columns.

- Summary. The name of the quantity computed from a trajectory or its observations.
- Definition. How the summary is computed from the sampled states and observations. The same computation runs on the available data.
- Data value. The value the summary takes on the available data, as a number or a range.
- Source. The dataset that provides the comparison, given with the archive identifier or repository link, or the paper or supplement passage when the value comes from there.

The check draws many trajectories, computes each summary on the draws and on the data, and asks whether the data summary sits within the spread of the simulated summaries. Choose summaries the data can speak to. Where no data constrains a summary, give a plausible range from the paper or the user's domain knowledge and name its source in the Data value cell.

For a figure recreation, name the figure and its locator in the paper, describe the quantity it shows, and state how the model reproduces it. The reproduction stands next to the original so the reader can compare them.

## Writing style

Write the PDF prose in the voice of a tenured machine learning professor. Keep it direct, technical, and clean of the usual tells of machine generation. Avoid semicolons, colons, and em-dashes. Avoid the construction that denies one claim to assert another, such as saying that something is not X but instead Y. Avoid trailing participial phrases of the form X does Y, enabling Z, and break them into separate sentences with explicit subjects and finite verbs. Avoid lists of negations of the form no X, no Y, no Z. Describe the model on its own terms. Do not reference an earlier draft of the writeup or justify the current text by contrast with what it replaced.


