"""pytest plugins that are loaded explicitly with ``-p``, never automatically.

A plugin in here changes the *shape of the environment* a run happens in, so it must be
asked for by name: the default ``pytest`` invocation has to be the one CI runs, and a plugin
that took effect just by existing would make that untrue for whoever checked the file out.
"""
