Notes: 

This file contains some example python code to find a beta distribution 
when given its confidence interval, e.g., it's 90%, 95% confidence interval. 

I am just presenting a very barebones implementation,
and I'm not producing the glue code to integrate this with the rest of Squigglepy.
In particular, if I move the file one folder up, I encounter this error:
[AttributeError: 'module' object has no attribute 'Number'](
https://stackoverflow.com/questions/38703637/attributeerror-module-object-has-no-attribute-number)

It does so by means of calling my personal server, 
which is running an instance of [this javascript package](https://www.npmjs.com/package/fit-beta). 
I am not making guarantees about it, 
but I don't intend to take it down in the near future either. 

The other possible approach would be to implement 
[Nelder Mead](https://en.wikipedia.org/wiki/Nelder%E2%80%93Mead_method),
or other such method within Python. 
I don't think it would be fundamentally difficult,
but it would take half a day to a few days worth of work, which feels wasteful.
For that approach, see the documentation for the fit-beta
[repository](https://github.com/quantified-uncertainty/fit-beta)
