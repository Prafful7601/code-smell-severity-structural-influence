"""
entity_finder.py

Cheap, regex + brace-matching heuristics for locating a Java class or
method declaration in a source file's text and extracting its body span,
WITHOUT a full parser. Used by 03_persistence.py to check, at each
subsequent revision of a file, whether an entity still exists and whether
a size/shape proxy for its smell still holds.

This is deliberately not a real parser (javalang/JDT). A full parser would
choke on any file using post-Java-8 syntax it doesn't support (var, records,
switch expressions, text blocks, sealed classes) and MLCQ's history spans
many such files; brace-counting degrades gracefully instead of failing
outright. Documented limitations (see METHOD.md):
  - Overload disambiguation uses parameter COUNT, not parameter TYPES.
  - A duplicate simple name in a different nested scope can be mismatched.
  - Braces inside string/char literals and comments are skipped, but a
    literal containing an unbalanced, unescaped quote can throw off the
    scanner for the rest of the file (rare in practice).
"""
import re
from dataclasses import dataclass
from typing import Optional

_LINE_COMMENT = "//"
_BLOCK_COMMENT_START = "/*"
_BLOCK_COMMENT_END = "*/"


@dataclass
class EntitySpan:
    start_line: int  # 1-indexed, line of the declaration
    end_line: int  # 1-indexed, line of the matching closing brace
    body: str


def _strip_comments_and_strings_mask(text: str) -> str:
    """Return a same-length string where chars inside string/char literals
    and comments are replaced with spaces (braces inside them neutralized),
    everything else preserved verbatim. Lets us brace-count on the mask
    while still slicing line numbers from the original text."""
    out = list(text)
    i, n = 0, len(text)
    in_line_comment = False
    in_block_comment = False
    in_string = False
    in_char = False
    while i < n:
        c = text[i]
        if in_line_comment:
            if c == "\n":
                in_line_comment = False
            else:
                out[i] = " "
        elif in_block_comment:
            if text[i:i + 2] == _BLOCK_COMMENT_END:
                out[i] = " "
                out[i + 1] = " "
                i += 2
                in_block_comment = False
                continue
            elif c != "\n":
                out[i] = " "
        elif in_string:
            if c == "\\":
                out[i] = " "
                if i + 1 < n:
                    out[i + 1] = " "
                i += 2
                continue
            if c == '"':
                in_string = False
            else:
                out[i] = " "
        elif in_char:
            if c == "\\":
                out[i] = " "
                if i + 1 < n:
                    out[i + 1] = " "
                i += 2
                continue
            if c == "'":
                in_char = False
            else:
                out[i] = " "
        else:
            if text[i:i + 2] == _LINE_COMMENT:
                in_line_comment = True
                out[i] = " "
                i += 1
            elif text[i:i + 2] == _BLOCK_COMMENT_START:
                in_block_comment = True
                out[i] = " "
                out[i + 1] = " "
                i += 2
                continue
            elif c == '"':
                in_string = True
            elif c == "'":
                in_char = True
        i += 1
    return "".join(out)


def _matching_brace_end_line(text: str, mask: str, open_brace_idx: int) -> Optional[int]:
    depth = 0
    for i in range(open_brace_idx, len(mask)):
        ch = mask[i]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text.count("\n", 0, i) + 1
    return None


def _find_open_brace_after(mask: str, from_idx: int) -> Optional[int]:
    # Find the first '{' after from_idx that isn't inside a comment/string
    # (already blanked in mask) -- but skip over a lambda-arrow '->{' vs
    # nested generic '<>' etc. is not an issue here since we scan raw chars.
    idx = mask.find("{", from_idx)
    return idx if idx != -1 else None


TYPE_DECL_RE_TMPL = r"\b(?:class|interface|enum|@\s*interface|record)\s+{name}\b"

# NOTE on an earlier version of this module: a prior implementation matched
# a method's return-type prefix with a regex like
#   (?:public|private|...|\s)* (?:<...>)? [\w\.\[\]<>,\s]+? \s+ name\s*\(
# Two adjacent quantified groups that both accept whitespace (the modifier
# alternation and the return-type character class) is a textbook catastrophic-
# backtracking shape: on a common name like "equals" appearing many times in
# a large file, Python's re engine can spend minutes on a single call. It was
# replaced with the linear, backtracking-free scan below (find the bare
# "name(" occurrences, then classify each candidate using plain string
# lookback/lookahead -- no regex over unbounded surrounding context).
_NAME_CALL_RE_CACHE: dict = {}


def _name_paren_matches(mask: str, name: str):
    """All positions where `name` is immediately followed by '(' (ignoring
    interior whitespace), found via a single literal regex with no nested
    quantifiers -- O(n), cannot backtrack pathologically."""
    pat = _NAME_CALL_RE_CACHE.get(name)
    if pat is None:
        pat = re.compile(r"\b" + re.escape(name) + r"\s*\(")
        _NAME_CALL_RE_CACHE[name] = pat
    return list(pat.finditer(mask))


def _looks_like_declaration(mask: str, name_start: int, paren_close: int) -> bool:
    """Cheap, backtracking-free heuristic distinguishing a declaration
    (`Type name(...) {`) from a call site (`obj.name(...)`, `name(...);`,
    a method reference, etc.), using only fixed-window string ops."""
    # not preceded (skipping whitespace) by '.' or '::' -- rules out
    # obj.name(...) calls and Type::name references
    j = name_start - 1
    while j >= 0 and mask[j] in " \t":
        j -= 1
    if j >= 0 and mask[j] in ".:":
        return False
    # must be immediately followed (after optional generic/throws clause) by
    # '{' before any ';' -- rules out call-site statements and abstract/
    # interface signatures with no body
    after = mask[paren_close + 1:paren_close + 300]
    semi_idx = after.find(";")
    brace_idx = after.find("{")
    if brace_idx == -1:
        return False
    if semi_idx != -1 and semi_idx < brace_idx:
        return False
    # only whitespace / 'throws ...' / generic bounds allowed between ')' and '{'
    between = after[:brace_idx]
    if not re.fullmatch(r"[\s\w.,<>\[\]]*", between):
        return False
    return True


def find_class_entity(source: str, simple_name: str) -> Optional[EntitySpan]:
    mask = _strip_comments_and_strings_mask(source)
    pattern = re.compile(TYPE_DECL_RE_TMPL.format(name=re.escape(simple_name)))
    m = pattern.search(mask)
    if not m:
        return None
    open_idx = _find_open_brace_after(mask, m.end())
    if open_idx is None:
        return None
    end_line = _matching_brace_end_line(source, mask, open_idx)
    if end_line is None:
        return None
    start_line = source.count("\n", 0, m.start()) + 1
    body = "\n".join(source.splitlines()[start_line - 1:end_line])
    return EntitySpan(start_line=start_line, end_line=end_line, body=body)


def _count_top_level_commas_plus_one(arglist: str) -> int:
    arglist = arglist.strip()
    if not arglist:
        return 0
    depth = 0
    count = 1
    for ch in arglist:
        if ch in "<([{":
            depth += 1
        elif ch in ">)]}":
            depth -= 1
        elif ch == "," and depth == 0:
            count += 1
    return count


def find_method_entity(source: str, simple_name: str, n_params: Optional[int]) -> Optional[EntitySpan]:
    """Find a method (or constructor) declaration by simple name, optionally
    disambiguating overloads by parameter COUNT (not type -- see module
    docstring). Returns the first candidate matching both name and arity;
    falls back to the first name-and-shape match if no arity match is found.
    Declaration vs. call-site is classified with fixed-window string checks
    (see _looks_like_declaration) rather than a regex return-type match, to
    stay linear-time even when `simple_name` (e.g. "equals", "toString") is
    a common token that appears hundreds of times in a large file."""
    mask = _strip_comments_and_strings_mask(source)

    def build_span(m):
        paren_open = mask.find("(", m.end() - 1)
        if paren_open == -1:
            return None, None
        depth = 0
        paren_close = None
        for i in range(paren_open, len(mask)):
            if mask[i] == "(":
                depth += 1
            elif mask[i] == ")":
                depth -= 1
                if depth == 0:
                    paren_close = i
                    break
        if paren_close is None:
            return None, None
        arglist = mask[paren_open + 1:paren_close]
        arity = _count_top_level_commas_plus_one(arglist)
        return paren_close, arity

    best = None
    for m in _name_paren_matches(mask, simple_name):
        paren_close, arity = build_span(m)
        if paren_close is None:
            continue
        if not _looks_like_declaration(mask, m.start(), paren_close):
            continue
        if n_params is not None and arity == n_params:
            best = (m, paren_close)
            break
        if best is None:
            best = (m, paren_close)  # fallback: first declaration-shaped match

    if best is None:
        return None
    m, paren_close = best
    open_idx = _find_open_brace_after(mask, paren_close)
    if open_idx is None:
        return None
    end_line = _matching_brace_end_line(source, mask, open_idx)
    if end_line is None:
        return None
    start_line = source.count("\n", 0, m.start()) + 1
    body = "\n".join(source.splitlines()[start_line - 1:end_line])
    return EntitySpan(start_line=start_line, end_line=end_line, body=body)


# ---- smell-condition proxies (operate on an EntitySpan.body) ----

# Keywords that can precede '(' but are not method declarations (control
# structures, operators). Excluded from the generic declaration scan below.
_NOT_A_METHOD_NAME = {
    "if", "for", "while", "switch", "catch", "synchronized", "return",
    "new", "instanceof", "throw", "assert", "do", "else", "try",
}

_ANY_NAME_PAREN_RE = re.compile(r"\b(\w+)\s*\(")  # linear -- single flat group, no ambiguity
_TRIVIAL_GETTER_RE = re.compile(r"\{\s*return\s+[\w.]+\s*;\s*\}")
_TRIVIAL_SETTER_RE = re.compile(r"\{\s*this\.\w+\s*=\s*\w+\s*;\s*\}")
_CONTROL_FLOW_RE = re.compile(r"\b(if|for|while|switch)\s*\(")
_INTERNAL_CALL_HINT_RE = re.compile(r"\bthis\.\w+")
_EXTERNAL_CALL_HINT_RE = re.compile(r"\b[a-z]\w*\.\w+\s*\(")  # var.method(


def _iter_method_decl_open_braces(body: str):
    """Yield the index (into `body`) of the opening '{' of every method-
    declaration-shaped construct in `body`, at any nesting depth. Same
    linear, backtracking-free classification as find_method_entity: a bare
    `name(` scan (single flat regex) filtered by _looks_like_declaration
    plus a denylist of control-flow keywords. `body` is assumed to already
    be comment/string-safe (an EntitySpan.body, sliced from a masked scan)."""
    mask = _strip_comments_and_strings_mask(body)
    for m in _ANY_NAME_PAREN_RE.finditer(mask):
        name = m.group(1)
        if name in _NOT_A_METHOD_NAME:
            continue
        paren_open = mask.find("(", m.end() - 1)
        if paren_open == -1:
            continue
        depth = 0
        paren_close = None
        for i in range(paren_open, len(mask)):
            if mask[i] == "(":
                depth += 1
            elif mask[i] == ")":
                depth -= 1
                if depth == 0:
                    paren_close = i
                    break
        if paren_close is None:
            continue
        if not _looks_like_declaration(mask, m.start(), paren_close):
            continue
        open_idx = _find_open_brace_after(mask, paren_close)
        if open_idx is not None:
            yield open_idx


def long_method_holds(span: EntitySpan, loc_threshold: int = 100) -> bool:
    loc = span.end_line - span.start_line + 1
    return loc >= loc_threshold


def blob_holds(span: EntitySpan, loc_threshold: int = 200, method_count_threshold: int = 20) -> bool:
    loc = span.end_line - span.start_line + 1
    # crude method count: declarations at any nesting depth within the body
    # (documented over-count risk if the class has non-static nested classes
    # with their own methods)
    n_methods = sum(1 for _ in _iter_method_decl_open_braces(span.body))
    return loc >= loc_threshold and n_methods >= method_count_threshold


def data_class_holds(span: EntitySpan, trivial_fraction_threshold: float = 0.70) -> bool:
    total = 0
    trivial = 0
    has_nontrivial_control_flow = False
    for open_idx in _iter_method_decl_open_braces(span.body):
        total += 1
        # look at the ~300 chars after the opening brace as a cheap body proxy
        snippet = span.body[open_idx + 1:open_idx + 1 + 300]
        if _TRIVIAL_GETTER_RE.search("{" + snippet) or _TRIVIAL_SETTER_RE.search("{" + snippet):
            trivial += 1
        if _CONTROL_FLOW_RE.search(snippet):
            has_nontrivial_control_flow = True
    if total == 0:
        return False
    frac = trivial / total
    return frac >= trivial_fraction_threshold and not has_nontrivial_control_flow


def feature_envy_holds(span: EntitySpan) -> bool:
    internal = len(_INTERNAL_CALL_HINT_RE.findall(span.body))
    external = len(_EXTERNAL_CALL_HINT_RE.findall(span.body))
    # exclude the trivial matches that are actually 'this.x' being counted
    # in both -- external regex requires lowercase var + '.', 'this' itself
    # won't match \b[a-z]\w*\. as a call because of the extra '(' requirement
    # differentiator is already handled by requiring '(' after external dot.
    return external > internal


SMELL_CONDITION_FN = {
    "long method": long_method_holds,
    "blob": blob_holds,
    "data class": data_class_holds,
    "feature envy": feature_envy_holds,
}
