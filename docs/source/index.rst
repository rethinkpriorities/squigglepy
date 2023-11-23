Squigglepy: Implementation of Squiggle in Python
================================================

`Squiggle <https://www.squiggle-language.com/>`__ is a "simple
programming language for intuitive probabilistic estimation". It serves
as its own standalone programming language with its own syntax, but it
is implemented in JavaScript. I like the features of Squiggle and intend
to use it frequently, but I also sometimes want to use similar
functionalities in Python, especially alongside other Python statistical
programming packages like Numpy, Pandas, and Matplotlib. The
**squigglepy** package here implements many Squiggle-like
functionalities in Python.

.. toctree::
   :maxdepth: 3
   :caption: Contents

   API Reference <reference/modules>
   Examples <examples>

Installation
------------

.. code:: shell

   pip install squigglepy

For plotting support, you can also use the ``plots`` extra:

.. code:: shell

   pip install squigglepy[plots]

Run tests
---------

Use ``black .`` for formatting.

Run
``ruff check . && pytest && pip3 install . && python3 tests/integration.py``

Disclaimers
-----------

This package is unofficial and supported by Peter Wildeford and Rethink
Priorities. It is not affiliated with or associated with the Quantified
Uncertainty Research Institute, which maintains the Squiggle language
(in JavaScript).

This package is also new and not yet in a stable production version, so
you may encounter bugs and other errors. Please report those so they can
be fixed. It’s also possible that future versions of the package may
introduce breaking changes.

This package is available under an MIT License.

Acknowledgements
----------------

-  The primary author of this package is Peter Wildeford. Agustín
   Covarrubias and Bernardo Baron contributed several key features and
   developments.
-  Thanks to Ozzie Gooen and the Quantified Uncertainty Research
   Institute for creating and maintaining the original Squiggle
   language.
-  Thanks to Dawn Drescher for helping me implement math between
   distributions.
-  Thanks to Dawn Drescher for coming up with the idea to use ``~`` as a
   shorthand for ``sample``, as well as helping me implement it.
