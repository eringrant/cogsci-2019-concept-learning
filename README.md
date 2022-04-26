# cogsci-2019-concept-learning

Code for our 2019 CogSci paper
["Learning deep taxonomic priors for concept learning from few positive examples."](https://cogsci.mindmodeling.org/2019/papers/0328/)
To cite the work that this code is associated with, use:

```
@inproceedings{grant2019learning,
  title={Learning deep taxonomic priors for concept learning from few positive examples},
  author={Grant, Erin and Peterson, Joshua C and Griffiths, Thomas L},
  booktitle={Proceedings of the Annual Conference of the Cognitive Science Society},
  year={2019}
}
```


## tl;dr

1. [Install the package](#installation). (Remember to `conda activate cogsci-2019-concept-learning` if necessary.)

1. Run the following commands to set up NLTK:

```shell
python -c "import nltk; nltk.download('wordnet'); nltk.download('omw-1.4')"
```

3. TODO(eringrant): Document how to get ImageNet images by synset.

4. A training and evaluation run on the human data can then be run via: 

```shell
scripts/run_human_comp.sh /tmp /tmp PATH_TO_IMAGENET
```


## Installation

### Option: Conda install

To install via [Conda](https://docs.conda.io/), do:

```shell
git clone git@github.com:eringrant/cogsci-2019-concept-learning.git
cd cogsci-2019-concept-learning
conda env create --file environment.yml
```

The Conda environment can then be activated via `conda activate cogsci-2019-concept-learning`.

### Option: pip install

To install via [pip](https://pip.pypa.io/), do:

```shell
git clone git@github.com:eringrant/cogsci-2019-concept-learning.git
cd cogsci-2019-concept-learning
pip install -e .
```
