=========
md_tester
=========

**Document format for Python/NumPy modules**

.. Is this a comment?  Yes it is ... you can't see me

.. _format:
.. contents:: **Table of Contents**
   :depth: 2

.. note::  This is a **note** box.  This one actually works 

.. --------------------------------------------------------------------------

.. From here down, gets pulled into the table of contents

Parameters
----------
No need for a blank line after the underlines, but you need one after this
line.

a : stuff
    A space before the colon only bold faces before it. The second line
    gets a four space indent and the text line wraps
b: more stuff
   But if you don't leave a space before the colon, the whole line gets
   bolded.
c : still more
   Now what 


.. --------------------------------------------------------------------------

Code samples
------------
Several examples of how to format code blocks, with and without line numbers.

.. line numbers with code block
.. code-block:: python
   :linenos:

   a = np.arange(5)
   print(a)

.. Code block removed.
.. code-block:: python

   a = np.arange(5)
   print(a)

.. --------------------------------------------------------------------------

Bullet lists
------------
Bullet lists need a blank line below and a blank line after

- ``double backticks`` : monospaced
- `single backticks`   : italics
- *one star*           : italics
- **two star**         : bold

So bullet is done.


Indentation extras
------------------
You can maintain indentation using vertical bars but the lines need to be
indented

| Vertical bars with at least 1 space before
| keep stuff lined up without needing a blank line

    | indentation and a blank line before
    | control the spaces before the bar

but as soon as it is gone, it wraps


Test section
------------
Now this block makes a table of contents from the first column and you
can put comments after the ( - )::

   .. autosummary::
      :toctree:

   arraytools.utils.time_deco - Runs timing functions
   arraytools.utils.run_deco  - Reports run information

That is the format, here is what is what it looks like.

.. autosummary::
   :toctree:

   ../../utils.time_deco

#   arraytools.utils.time_deco - Runs timing functions
#   arraytools.utils.run_deco  - Reports run information
#   arraytools.utils.get_func  - Retrieves function information
#   arraytools.utils.get_modu  - Retrieves module information

.. warning:: cant see me

Between the table and this line is a hidden warning.  It takes the form
::

    ..warning:: can't see me

Some text for warnings you can't see me because dot dot is a comment


.. seealso:: This is a simple **seealso** note.

.. hlist::
    :columns: 3

    * hlist needed
    * so is columns
    * 3 is the number of columns
    * this is the last of col 2
    * and column 3 straggler

Another section
---------------
In Test Section, we made a block of function names followed by a comment.
If we leave the comment out, the documentation from that function is returned::

    .. autosummary::
       :toctree:
    
       ../../utils

It appears like this (with several more shown).

.. autosummary::
   :toctree:

   ../../utils

#   arraytools.utils.time_deco
#   arraytools.utils.run_deco
#   arraytools.utils.get_func
#   arraytools.utils.get_modu
#   arraytools.utils.dirr

The `utils` module is cool.

