# Vertex Reconstruction Overview

When antihydrogen annihilates, it typically produces 2-4 charged pions, which
travel through the detector and leave behind a trail of hits.

The challenge is to infer the **origin point** of these tracks i.e. the location
of the annihilation.

In traditional reconstruction pipelines, this is done using geometry-based
algorithms that:
- Identify clusters of hits,
- Attempt to form tracks,
- Fit back to a common origin.

In this tutorial, you'll work with a **deep learning-based approach**:
- We provide you with a working model that predicts the vertex directly from a
  cloud of hits.
- You'll explore how this model works and then experiment with ways to make it
  better.

By the end of the tutorial, you'll have applied meaningful improvements to a
real-world ML system used in fundamental physics research.
