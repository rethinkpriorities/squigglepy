## Squigglepy: Implementation of Squiggle in Python

[Squiggle](https://www.squiggle-language.com/) is a "simple programming language for intuitive probabilistic estimation". It serves as its own standalone programming language with its own syntax, but it is implemented in JavaScript. I like the features of Squiggle and intend to use it frequently, but I also sometimes want to use similar functionalities in Python, especially alongside other Python statistical programming packages like Numpy, Pandas, and Matplotlib. The **squigglepy** package here implements many Squiggle-like functionalities in Python.


## Usage

Here's the Squigglepy implementation of [the example from Squiggle Docs](https://www.squiggle-language.com/docs/Overview):

```Python
from squigglepy.sample import *
from squigglepy.utils import *
from squigglepy.numbers import *

populationOfNewYork2022 = to(8.1*million(), 8.4*million()) # This means that you're 90% confident the value is between 8.1 and 8.4 Million.

def proportionOfPopulationWithPianos():
    percentage = to(.2, 1)
    return sample(percentage) * 0.01 # We assume there are almost no people with multiple pianos

def pianoTunersPerPiano():
    pianosPerPianoTuner = to(2*thousand(), 50*thousand())
    return 1 / sample(pianosPerPianoTuner)

def totalTunersIn2022():
    return (sample(populationOfNewYork2022) *
            proportionOfPopulationWithPianos() *
            pianoTunersPerPiano())

get_percentiles(sample(totalTunersIn2022, n=1000))
```

And the version from the Squiggle doc that incorporates time:

```Python
from squigglepy.sample import *
from squigglepy.utils import *
from squigglepy.numbers import *

populationOfNewYork2022 = to(8.1*million(), 8.4*million())

def proportionOfPopulationWithPianos():
    percentage = to(.2, 1)
    return sample(percentage) * 0.01

def proportionOfPopulationWithPianos():
    percentage = to(.2, 1)
    return sample(percentage) * 0.01

def pianoTunersPerPiano():
    pianosPerPianoTuner = to(2*thousand(), 50*thousand())
    return 1 / sample(pianosPerPianoTuner)

# Time in years after 2022
def populationAtTime(t):
    averageYearlyPercentageChange = to(-0.01, 0.05) # We're expecting NYC to continuously grow with an mean of roughly between -1% and +4% per year
    return sample(populationOfNewYork2022) * ((sample(averageYearlyPercentageChange) + 1) ** t)
}

def totalTunersAtTime(t):
	  return (populationAtTime(t) *
            proportionOfPopulationWithPianos() *
            pianoTunersPerPiano())

get_percentiles(sample(lambda: totalTunersAtTime(2030-2022), n=1000))
```

## Installation

`pip3 install squigglepy`

