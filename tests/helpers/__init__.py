"""Plain helper functions that several test modules share.

Fixtures belong in ``tests/conftest.py``; these are ordinary functions, imported
explicitly the way the modules import ``conftest`` and ``browser_sim``
(``tests/`` is on ``sys.path`` while pytest runs):

    from helpers.apptest import all_markdown

* ``helpers.apptest`` - text readers for a Streamlit ``AppTest`` page and the one
  Decision 4 app script that two modules drive.
* ``helpers.d4_reference`` - a row of a Decision 4 reference extract
  (``data/stata_d4_verification.csv`` / ``data/stata_d4_flexibility_verification.csv``)
  as the agent state the decision functions take.

Only helpers that more than one module used to carry its own copy of live here.
"""
