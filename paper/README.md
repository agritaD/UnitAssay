# paper/

Code that regenerates every measured panel in the unitassay paper, from a
curated corpus. Nothing here is imported by the package; it is a consumer of
`unitassay`'s output, so the library stays free of figure and path code.

Each module is named for the figure it makes, and each **data** function is
named for the quantity it regenerates, so a reader can run one function and get
one number back rather than having to run a whole figure.

    layer1_screen.py    the screen layer: channel selection, thresholds,
                        performance against the sorter's own label, cost

Every data function takes the corpus root as an argument and returns a plain
dict, so it can be called without matplotlib and its output diffed between
runs. `make_figure()` only draws; it never measures.
