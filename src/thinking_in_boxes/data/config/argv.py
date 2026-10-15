# Lenient argv scanning for the scripts BlenderProc runs.
#
# ``blenderproc run`` hands the script an argv containing its own arguments as
# well, so these scripts scan for the flags they know and ignore everything
# else rather than erroring on unknown tokens the way argparse would.
#
# Declare options with :func:`value` and :func:`flag`; :func:`parse_argv`
# returns only the keys that were actually present, so callers can tell "not
# given" from "given the default".

_VALUE = "value"
_FLAG = "flag"


def value(name, cast=str):
    """An option that consumes the next argv entry, e.g. ``--count 10``."""
    return (_VALUE, name, cast)


def flag(name, constant=True):
    """A standalone switch, e.g. ``--fast``."""
    return (_FLAG, name, constant)


def parse_argv(argv, options):
    """Scan *argv* for *options* ({token: value(...)/flag(...)}).

    Unknown tokens are skipped, as is a value option in final position with
    nothing following it.  Later occurrences win.
    """
    parsed = {}
    i = 0
    while i < len(argv):
        spec = options.get(argv[i])
        if spec is None:
            i += 1
            continue
        kind, name, arg = spec
        if kind == _FLAG:
            parsed[name] = arg
            i += 1
        elif i + 1 < len(argv):
            parsed[name] = arg(argv[i + 1])
            i += 2
        else:
            i += 1
    return parsed
