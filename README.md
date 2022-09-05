## Squigglepy: Implementation of Squiggle in Python

[Squiggle](https://www.squiggle-language.com/) is a "simple programming language for intuitive probabilistic estimation". It serves as its own standalone programming language with its own syntax, but it is implemented in JavaScript. I like the features of Squiggle and intend to use it frequently, but I also sometimes want to use similar functionalities in Python, especially alongside other Python statistical programming packages like Numpy, Pandas, and Matplotlib. The **squigglepy** package here implements many Squiggle-like functionalities in Python.


## Usage

Here's the Squigglepy implementation of [the example from Squiggle Docs](https://www.squiggle-language.com/docs/Overview):

```Python
import squigglepy as sq

populationOfNewYork2022 = sq.to(8.1*million(), 8.4*million()) # This means that you're 90% confident the value is between 8.1 and 8.4 Million.

def proportionOfPopulationWithPianos():
    percentage = sq.to(.2, 1)
    return sq.sample(percentage) * 0.01 # We assume there are almost no people with multiple pianos

def pianoTunersPerPiano():
    pianosPerPianoTuner = sq.to(2*thousand(), 50*thousand())
    return 1 / sq.sample(pianosPerPianoTuner)

def totalTunersIn2022():
    return (sq.sample(populationOfNewYork2022) *
            proportionOfPopulationWithPianos() *
            pianoTunersPerPiano())

sq.get_percentiles(sq.sample(totalTunersIn2022, n=1000))
```

And the version from the Squiggle doc that incorporates time:

```Python
import squigglepy as sq
K = sq.thousand(); M = sq.million()

populationOfNewYork2022 = sq.to(8.1*M, 8.4*M)

def proportionOfPopulationWithPianos():
    percentage = sq.to(.2, 1)
    return sq.sample(percentage) * 0.01

def proportionOfPopulationWithPianos():
    percentage = sq.to(.2, 1)
    return sq.sample(percentage) * 0.01

def pianoTunersPerPiano():
    pianosPerPianoTuner = sq.to(2*K, 50*K)
    return 1 / sq.sample(pianosPerPianoTuner)

# Time in years after 2022
def populationAtTime(t):
    averageYearlyPercentageChange = sq.to(-0.01, 0.05) # We're expecting NYC to continuously grow with an mean of roughly between -1% and +4% per year
    return sq.sample(populationOfNewYork2022) * ((sq.sample(averageYearlyPercentageChange) + 1) ** t)

def totalTunersAtTime(t):
    return (populationAtTime(t) *
            proportionOfPopulationWithPianos() *
            pianoTunersPerPiano())

sq.get_percentiles(sq.sample(lambda: totalTunersAtTime(2030-2022), n=1000))
```

## Additional Features

Additional distributions:

```Python
import squigglepy as sq

# Normal distribution
sq.sample(sq.norm(1, 3))  # 90% interval from 1 to 3

# Distribution can be sampled with mean and sd too
sq.sample(sq.norm(mean=2, sd=3))

# Other distributions exist
sq.sample(sq.lognorm(1, 10))
sq.sample(sq.tdist(1, 10, t=5))

# You can mix distributions together
sq.sample(sq.mixture([sq.norm(1, 3),
                      sq.norm(4, 10),
                      sq.lognorm(1, 10)],  # Distributions to mix
                     [0.3, 0.3, 0.4]))     # These are the weights on each distribution

# You can change the CI from 90% (default) to 80%
sq.sample(sq.norm(1, 3, credibility=0.8))
```

## Installation

`pip3 install squigglepy`

