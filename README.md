# brainhack-2025

Repo for Pre-Cosyne brainhack 2025 project "Create notebooks importing published models in plenoptic".

## Overview (from google doc)

[plenoptic](https://plenoptic.org/) is a python library (built with pytorch) for model-based synthesis of perceptual stimuli, intended for researchers in neuroscience, psychology, and machine learning. The stimuli generated by plenoptic enable interpretation of model properties through features that are enhanced, suppressed, or discarded. Plenoptic contains a variety of vision science models (e.g., simple models of the retinal ganglion cells, the Portilla-Simoncelli texture model), but the synthesis methods can be used with any pytorch model, including those developed by the machine learning community. Currently, the plenoptic documentation mainly shows how to use the synthesis methods with the package-internal models. For this brainhack, we would like to develop notebooks that demonstrate how to take models from existing model zoos (e.g., [BrainScore](https://www.brain-score.org/tutorials/models/quickstart), [HuggingFace](https://huggingface.co/models?pipeline_tag=image-classification&sort=trending), [Torchvision](https://pytorch.org/vision/stable/models.html), [RobustBench](https://github.com/RobustBench/robustbench?tab=readme-ov-file#model-zoo-quick-tour)) and use them with plenoptic’s methods. These models must be: written in pytorch with a forward method, accept 4d tensors of images as input, and return a 3d or 4d tensor as output (see [plenoptic docs](https://docs.plenoptic.org/docs/branch/main/models.html) for more details on model requirements). Additionally, a common use case is to take, not the final model output, but those of several intermediate layers, so we should demonstrate the easiest way to do this, potentially using [torchvision’s feature extractor](https://pytorch.org/vision/stable/feature_extraction.html). Specific examples could include: synthesizing [metamers](https://docs.plenoptic.org/docs/branch/main/tutorials/intro/06_Metamer.html) for the leading V1 model on BrainScore, using the Metamer class to perform style transfer (Gatys et al, 2015, with VGG19). Finding other relevant model zoos (potentially from competitions at Cosyne or NeurIPS) would also be useful.

## Goals

1. Find model zoos that will be useful for users of plenoptic:
   - [Torchvision](https://pytorch.org/vision/stable/models.html)
   - [BrainScore](https://www.brain-score.org/tutorials/models/quickstart)
   - [HuggingFace](https://huggingface.co/models?pipeline_tag=image-classification&sort=trending)
   - [RobustBench](https://github.com/RobustBench/robustbench?tab=readme-ov-file#model-zoo-quick-tour)
2. Document how to use their models with plenoptic. See `plenoptic-example.ipynb` to see the minimal steps we want to be able to recreate.
3. Find specific interesting / relevant models from each model zoo, with citations.
4. Bonus ideas:
    - Write a function that uses the brainscore API to get the current Nth best neural models for specific brain areas and prepares it for plenoptic.
    - Use plenoptic's `Metamer` class to perform style transfer as in Gatys et al. 2015 (which used VGG-19).
    - Use plenotic's `MADCompetition` class to compare against among brainscore's top models for different brain areas, or between the best neural and behavioral models.

## Setup 

- Give Billy your github username so you can be given write access to this repo.
- Follow the instructions below to create your environment or login to the binder.
- If local, you will need to clone this git repo; the binder instance already has a cloned copy.
- Look at the `plenoptic-example.ipynb` notebook, to see a small example of how to use plenoptic, and the steps we'll want to be able to perform with the new models. Please don't push any changes to this file!
- Work in a jupyter notebook in the root directory of this repo, and make sure to push to origin before you leave!

### Local

Create a virtual environment with python 3.11 using your virtual environment manager of choice and then install the requirements found in `requirements.txt` by, e.g., running `pip install -r requirements.txt`. This environment includes: torch, torchvision, plenoptic, brainscore, timm, robustbench, and jupyter. If you think additional model zoos would be helpful, let Billy know!

While plenotic and torch are on conda-forge, I do not think the various model zoos are.

> [!WARNING]
> To generate the movies in plenoptic, you will also need `ffmpeg`. You may already have it installed on your machine, otherwise you can install it from [the official website](https://ffmpeg.org/download.html) or [conda-forge](https://anaconda.org/conda-forge/ffmpeg).

### Binder

[![Binder](https://mybinder.org/badge_logo.svg)](https://binder.flatironinstitute.org/v2/user/wbroderick/brainhack2025)

Click on the binder badge above to launch the binder instance, which has a GPU available. You will need to give Billy your email address before you do.

After doing so, we'll need to create an ssh key within the binder and add that to your GitHub, so you can push:

```sh
ssh-keygen -t rsa -f ~/.ssh/id_rsa
# for some reason, the Docker container doesn't have an ssh agent running by default
eval `ssh-agent`
ssh-add ~/.ssh/id_rsa
```

Then add the resulting `~/.ssh/id_rsa.pub` to your GitHub SSH keys. I would remember to delete that at the end of this hackathon, just to be safe.

I believe you'll need to run the `eval` and `ssh-add` lines each time your binder server starts up.

You'll also need to configure git with your username and email (you'll be prompted to do this the first time you try to commit):

``` sh
git config --global user.name "YOUR NAME HERE"
git config --global user.email "yourname@domain.com"
```

Some usage notes:

- You are only allowed to have a single binder instance running at a time, so if you get the "already have an instance running error", go to the [binderhub page](https://binder.flatironinstitute.org/hub/hub/home) (or click on "check your currently running servers" on the right of the page) to join your running instance.
- If you lose connection at any point, go to the [binderhub page](https://binder.flatironinstitute.org/hub/hub/home) to join your running instance rather than restarting the image.
- The binder will be shutdown automatically after 1 day of inactivity, so save everything before leaving for the day.
