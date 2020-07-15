import sys
import string
from six import string_types, text_type
import re
from collections import defaultdict


from nltk.tree import Tree
'''
def unicode_repr(obj):
    PY3 = sys.version_info[0] == 3
    if PY3:
        return repr(obj)

    if hasattr(obj, 'unicode_repr'):
        return obj.unicode_repr()

    if isinstance(obj, text_type):
        return repr(obj)[1:]  # strip "u" letter from output

    return repr(obj)


class Nonterminal(object):

    def __init__(self, symbol):

        self._symbol = symbol
        self._hash = hash(symbol)

    def symbol(self):

        return self._symbol

    def __eq__(self, other):

        return type(self) == type(other) and self._symbol == other._symbol

    def __ne__(self, other):
        return not self == other

    # def __lt__(self, other):
    #     if not isinstance(other, Nonterminal):
    # 		raise_unorderable_types("<", self, other)
    #     return self._symbol < other._symbol

    def __hash__(self):
        return self._hash

    def __repr__(self):

        if isinstance(self._symbol, string_types):
            return '%s' % self._symbol
        else:
            return '%s' % unicode_repr(self._symbol)

    def __str__(self):

        if isinstance(self._symbol, string_types):
            return '%s' % self._symbol
        else:
            return '%s' % unicode_repr(self._symbol)

    def __div__(self, rhs):
        """
        Return a new nonterminal whose symbol is ``A/B``, where ``A`` is
        the symbol for this nonterminal, and ``B`` is the symbol for rhs.

        :param rhs: The nonterminal used to form the right hand side
            of the new nonterminal.
        :type rhs: Nonterminal
        :rtype: Nonterminal
        """
        return Nonterminal('%s/%s' % (self._symbol, rhs._symbol))

    def __truediv__(self, rhs):
        """
        Return a new nonterminal whose symbol is ``A/B``, where ``A`` is
        the symbol for this nonterminal, and ``B`` is the symbol for rhs.
        This function allows use of the slash ``/`` operator with
        the future import of division.

        :param rhs: The nonterminal used to form the right hand side
            of the new nonterminal.
        :type rhs: Nonterminal
        :rtype: Nonterminal
        """
        return self.__div__(rhs)


class Production(object):

    def __init__(self, lhs, rhs):

        if isinstance(rhs, string_types):
            raise TypeError('production right hand side should be a list, '
                            'not a string')
        self._lhs = lhs
        self._rhs = tuple(rhs)
        self._hash = hash((self._lhs, self._rhs))

    def lhs(self):

        return self._lhs

    def rhs(self):

        return self._rhs

    def __len__(self):

        return len(self._rhs)

    def is_nonlexical(self):

        return all(is_nonterminal(n) for n in self._rhs)

    def is_lexical(self):

        return not self.is_nonlexical()

    def __str__(self):

        result = '%s -> ' % unicode_repr(self._lhs)
        result += " ".join(unicode_repr(el) for el in self._rhs)
        return result

    def __repr__(self):

        return '%s' % self

    def __eq__(self, other):

        return (type(self) == type(other) and
                self._lhs == other._lhs and
                self._rhs == other._rhs)

    def __ne__(self, other):
        return not self == other

    def __lt__(self, other):
        if not isinstance(other, Production):
            def raise_unorderable_types(ordering, a, b):
                raise TypeError("unorderable types: %s() %s %s()" % (type(a).__name__, ordering, type(b).__name__))
        raise_unorderable_types("<", self, other)
        return (self._lhs, self._rhs) < (other._lhs, other._rhs)

    def __hash__(self):
        """
        Return a hash value for the ``Production``.

        :rtype: int
        """
        return self._hash


class Tree(list):

    def __init__(self, node, children=None):
        if children is None:
            raise TypeError("%s: Expected a node value and child list "
                            % type(self).__name__)
        elif isinstance(children, string_types):
            raise TypeError("%s() argument 2 should be a list, not a "
                            "string" % type(self).__name__)
        else:
            list.__init__(self, children)
            self._label = node


    def __eq__(self, other):
        return (self.__class__ is other.__class__ and
                (self._label, list(self)) == (other._label, list(other)))

    def __lt__(self, other):
        if not isinstance(other, Tree):
            # raise_unorderable_types("<", self, other)
            # Sometimes children can be pure strings,
            # so we need to be able to compare with non-trees:
            return self.__class__.__name__ < other.__class__.__name__
        elif self.__class__ is other.__class__:
            return (self._label, list(self)) < (other._label, list(other))
        else:
            return self.__class__.__name__ < other.__class__.__name__

    __ne__ = lambda self, other: not self == other
    __gt__ = lambda self, other: not (self < other or self == other)
    __le__ = lambda self, other: self < other or self == other
    __ge__ = lambda self, other: not self < other

    def __mul__(self, v):
        raise TypeError('Tree does not support multiplication')

    def __rmul__(self, v):
        raise TypeError('Tree does not support multiplication')

    def __add__(self, v):
        raise TypeError('Tree does not support addition')

    def __radd__(self, v):
        raise TypeError('Tree does not support addition')

    def __getitem__(self, index):
        if isinstance(index, (int, slice)):
            return list.__getitem__(self, index)
        elif isinstance(index, (list, tuple)):
            if len(index) == 0:
                return self
            elif len(index) == 1:
                return self[index[0]]
            else:
                return self[index[0]][index[1:]]
        else:
            raise TypeError("%s indices must be integers, not %s" %
                            (type(self).__name__, type(index).__name__))

    def __setitem__(self, index, value):
        if isinstance(index, (int, slice)):
            return list.__setitem__(self, index, value)
        elif isinstance(index, (list, tuple)):
            if len(index) == 0:
                raise IndexError('The tree position () may not be '
                                 'assigned to.')
            elif len(index) == 1:
                self[index[0]] = value
            else:
                self[index[0]][index[1:]] = value
        else:
            raise TypeError("%s indices must be integers, not %s" %
                            (type(self).__name__, type(index).__name__))

    def __delitem__(self, index):
        if isinstance(index, (int, slice)):
            return list.__delitem__(self, index)
        elif isinstance(index, (list, tuple)):
            if len(index) == 0:
                raise IndexError('The tree position () may not be deleted.')
            elif len(index) == 1:
                del self[index[0]]
            else:
                del self[index[0]][index[1:]]
        else:
            raise TypeError("%s indices must be integers, not %s" %
                            (type(self).__name__, type(index).__name__))

    def _get_node(self):
        raise NotImplementedError("Use label() to access a node label.")

    def _set_node(self, value):
        raise NotImplementedError("Use set_label() method to set a node label.")

    node = property(_get_node, _set_node)

    def label(self):

        return self._label

    def set_label(self, label):

        self._label = label

    def leaves(self):

        leaves = []
        for child in self:
            if isinstance(child, Tree):
                leaves.extend(child.leaves())
            else:
                leaves.append(child)
        return leaves

    def flatten(self):

        return Tree(self.label(), self.leaves())

    def height(self):

        max_child_height = 0
        for child in self:
            if isinstance(child, Tree):
                max_child_height = max(max_child_height, child.height())
            else:
                max_child_height = max(max_child_height, 1)
        return 1 + max_child_height

    def treepositions(self, order='preorder'):

        positions = []
        if order in ('preorder', 'bothorder'): positions.append(())
        for i, child in enumerate(self):
            if isinstance(child, Tree):
                childpos = child.treepositions(order)
                positions.extend((i,) + p for p in childpos)
            else:
                positions.append((i,))
        if order in ('postorder', 'bothorder'): positions.append(())
        return positions

    def subtrees(self, filter=None):

        if not filter or filter(self):
            yield self
        for child in self:
            if isinstance(child, Tree):
                for subtree in child.subtrees(filter):
                    yield subtree

    def productions(self):

        if not isinstance(self._label, string_types):
            raise TypeError('Productions can only be generated from trees having node labels that are strings')

        prods = [Production(Nonterminal(self._label), _child_names(self))]
        for child in self:
            if isinstance(child, Tree):
                prods += child.productions()
        return prods

    def pos(self):

        pos = []
        for child in self:
            if isinstance(child, Tree):
                pos.extend(child.pos())
            else:
                pos.append((child, self._label))
        return pos

    def leaf_treeposition(self, index):

        if index < 0: raise IndexError('index must be non-negative')

        stack = [(self, ())]
        while stack:
            value, treepos = stack.pop()
            if not isinstance(value, Tree):
                if index == 0:
                    return treepos
                else:
                    index -= 1
            else:
                for i in range(len(value) - 1, -1, -1):
                    stack.append((value[i], treepos + (i,)))

        raise IndexError('index must be less than or equal to len(self)')

    def treeposition_spanning_leaves(self, start, end):

        if end <= start:
            raise ValueError('end must be greater than start')
        start_treepos = self.leaf_treeposition(start)
        end_treepos = self.leaf_treeposition(end - 1)
        for i in range(len(start_treepos)):
            if i == len(end_treepos) or start_treepos[i] != end_treepos[i]:
                return start_treepos[:i]
        return start_treepos

    def chomsky_normal_form(self, factor="right", horzMarkov=None, vertMarkov=0, childChar="|", parentChar="^"):

        from nltk.treetransforms import chomsky_normal_form
        chomsky_normal_form(self, factor, horzMarkov, vertMarkov, childChar, parentChar)

    def un_chomsky_normal_form(self, expandUnary=True, childChar="|", parentChar="^", unaryChar="+"):

        from nltk.treetransforms import un_chomsky_normal_form
        un_chomsky_normal_form(self, expandUnary, childChar, parentChar, unaryChar)

    def collapse_unary(self, collapsePOS=False, collapseRoot=False, joinChar="+"):

        from nltk.treetransforms import collapse_unary
        collapse_unary(self, collapsePOS, collapseRoot, joinChar)

    @classmethod
    def convert(cls, tree):

        if isinstance(tree, Tree):
            children = [cls.convert(child) for child in tree]
            return cls(tree._label, children)
        else:
            return tree

    def copy(self, deep=False):
        if not deep:
            return type(self)(self._label, self)
        else:
            return type(self).convert(self)

    def _frozen_class(self):
        return ImmutableTree

    def freeze(self, leaf_freezer=None):
        frozen_class = self._frozen_class()
        if leaf_freezer is None:
            newcopy = frozen_class.convert(self)
        else:
            newcopy = self.copy(deep=True)
            for pos in newcopy.treepositions('leaves'):
                newcopy[pos] = leaf_freezer(newcopy[pos])
            newcopy = frozen_class.convert(newcopy)
        hash(newcopy)  # Make sure the leaves are hashable.
        return newcopy

    @classmethod
    def fromstring(cls, s, brackets='()', read_node=None, read_leaf=None,
                   node_pattern=None, leaf_pattern=None,
                   remove_empty_top_bracketing=False):

        if not isinstance(brackets, string_types) or len(brackets) != 2:
            raise TypeError('brackets must be a length-2 string')
        if re.search('\s', brackets):
            raise TypeError('whitespace brackets not allowed')
        # Construct a regexp that will tokenize the string.
        open_b, close_b = brackets
        open_pattern, close_pattern = (re.escape(open_b), re.escape(close_b))
        if node_pattern is None:
            node_pattern = '[^\s%s%s]+' % (open_pattern, close_pattern)
        if leaf_pattern is None:
            leaf_pattern = '[^\s%s%s]+' % (open_pattern, close_pattern)
        token_re = re.compile('%s\s*(%s)?|%s|(%s)' % (
            open_pattern, node_pattern, close_pattern, leaf_pattern))
        # Walk through each token, updating a stack of trees.
        stack = [(None, [])]  # list of (node, children) tuples
        for match in token_re.finditer(s):
            token = match.group()
            # Beginning of a tree/subtree
            if token[0] == open_b:
                if len(stack) == 1 and len(stack[0][1]) > 0:
                    cls._parse_error(s, match, 'end-of-string')
                label = token[1:].lstrip()
                if read_node is not None: label = read_node(label)
                stack.append((label, []))
            # End of a tree/subtree
            elif token == close_b:
                if len(stack) == 1:
                    if len(stack[0][1]) == 0:
                        cls._parse_error(s, match, open_b)
                    else:
                        cls._parse_error(s, match, 'end-of-string')
                label, children = stack.pop()
                stack[-1][1].append(cls(label, children))
            # Leaf node
            else:
                if len(stack) == 1:
                    cls._parse_error(s, match, open_b)
                if read_leaf is not None: token = read_leaf(token)
                stack[-1][1].append(token)

        # check that we got exactly one complete tree.
        if len(stack) > 1:
            cls._parse_error(s, 'end-of-string', close_b)
        elif len(stack[0][1]) == 0:
            cls._parse_error(s, 'end-of-string', open_b)
        else:
            assert stack[0][0] is None
            assert len(stack[0][1]) == 1
        tree = stack[0][1][0]

        # If the tree has an extra level with node='', then get rid of
        # it.  E.g.: "((S (NP ...) (VP ...)))"
        if remove_empty_top_bracketing and tree._label == '' and len(tree) == 1:
            tree = tree[0]
        # return the tree.
        return tree

    @classmethod
    def _parse_error(cls, s, match, expecting):
        if match == 'end-of-string':
            pos, token = len(s), 'end-of-string'
        else:
            pos, token = match.start(), match.group()
        msg = '%s.read(): expected %r but got %r\n%sat index %d.' % (
            cls.__name__, expecting, token, ' ' * 12, pos)
        # Add a display showing the error token itsels:
        s = s.replace('\n', ' ').replace('\t', ' ')
        offset = pos
        if len(s) > pos + 10:
            s = s[:pos + 10] + '...'
        if pos > 10:
            s = '...' + s[pos - 10:]
            offset = 13
        msg += '\n%s"%s"\n%s^' % (' ' * 16, s, ' ' * (17 + offset))
        raise ValueError(msg)

    def draw(self):
        from nltk.draw.tree import draw_trees
        draw_trees(self)

    def pretty_print(self, sentence=None, highlight=(), stream=None, **kwargs):

        from nltk.treeprettyprinter import TreePrettyPrinter
        print(TreePrettyPrinter(self, sentence, highlight).text(**kwargs),
              file=stream)

    def __repr__(self):
        childstr = ", ".join(unicode_repr(c) for c in self)
        return '%s(%s, [%s])' % (type(self).__name__, unicode_repr(self._label), childstr)

    def _repr_png_(self):

        import os
        import base64
        import subprocess
        import tempfile
        from nltk.draw.tree import tree_to_treesegment
        from nltk.draw.util import CanvasFrame
        from nltk.internals import find_binary
        _canvas_frame = CanvasFrame()
        widget = tree_to_treesegment(_canvas_frame.canvas(), self)
        _canvas_frame.add_widget(widget)
        x, y, w, h = widget.bbox()
        # print_to_file uses scrollregion to set the width and height of the pdf.
        _canvas_frame.canvas()['scrollregion'] = (0, 0, w, h)
        with tempfile.NamedTemporaryFile() as file:
            in_path = '{0:}.ps'.format(file.name)
            out_path = '{0:}.png'.format(file.name)
            _canvas_frame.print_to_file(in_path)
            _canvas_frame.destroy_widget(widget)
            subprocess.call([find_binary('gs', binary_names=['gswin32c.exe', 'gswin64c.exe'], env_vars=['PATH'], verbose=False)] +
                            '-q -dEPSCrop -sDEVICE=png16m -r90 -dTextAlphaBits=4 -dGraphicsAlphaBits=4 -dSAFER -dBATCH -dNOPAUSE -sOutputFile={0:} {1:}'
                            .format(out_path, in_path).split())
            with open(out_path, 'rb') as sr:
                res = sr.read()
            os.remove(in_path)
            os.remove(out_path)
            return base64.b64encode(res).decode()

    def __str__(self):
        return self.pformat()

    def pprint(self, **kwargs):

        if "stream" in kwargs:
            stream = kwargs["stream"]
            del kwargs["stream"]
        else:
            stream = None
        print(self.pformat(**kwargs), file=stream)

    def pformat(self, margin=70, indent=0, nodesep='', parens='()', quotes=False):

        s = self._pformat_flat(nodesep, parens, quotes)
        if len(s) + indent < margin:
            return s

        if isinstance(self._label, string_types):
            s = '%s%s%s' % (parens[0], self._label, nodesep)
        else:
            s = '%s%s%s' % (parens[0], unicode_repr(self._label), nodesep)
        for child in self:
            if isinstance(child, Tree):
                s += '\n' + ' ' * (indent + 2) + child.pformat(margin, indent + 2,
                                                               nodesep, parens, quotes)
            elif isinstance(child, tuple):
                s += '\n' + ' ' * (indent + 2) + "/".join(child)
            elif isinstance(child, string_types) and not quotes:
                s += '\n' + ' ' * (indent + 2) + '%s' % child
            else:
                s += '\n' + ' ' * (indent + 2) + unicode_repr(child)
        return s + parens[1]

    def pformat_latex_qtree(self):

        reserved_chars = re.compile('([#\$%&~_\{\}])')

        pformat = self.pformat(indent=6, nodesep='', parens=('[.', ' ]'))
        return r'\Tree ' + re.sub(reserved_chars, r'\\\1', pformat)

    def _pformat_flat(self, nodesep, parens, quotes):
        childstrs = []
        for child in self:
            if isinstance(child, Tree):
                childstrs.append(child._pformat_flat(nodesep, parens, quotes))
            elif isinstance(child, tuple):
                childstrs.append("/".join(child))
            elif isinstance(child, string_types) and not quotes:
                childstrs.append('%s' % child)
            else:
                childstrs.append(unicode_repr(child))
        if isinstance(self._label, string_types):
            return '%s%s%s %s%s' % (parens[0], self._label, nodesep,
                                    " ".join(childstrs), parens[1])
        else:
            return '%s%s%s %s%s' % (parens[0], unicode_repr(self._label), nodesep,
                                    " ".join(childstrs), parens[1])

class ImmutableTree(Tree):
    def __init__(self, node, children=None):
        super(ImmutableTree, self).__init__(node, children)
        # Precompute our hash value.  This ensures that we're really
        # immutable.  It also means we only have to calculate it once.
        try:
            self._hash = hash((self._label, tuple(self)))
        except (TypeError, ValueError):
            raise ValueError("%s: node value and children "
                             "must be immutable" % type(self).__name__)

    def __setitem__(self, index, value):
        raise ValueError('%s may not be modified' % type(self).__name__)
    def __setslice__(self, i, j, value):
        raise ValueError('%s may not be modified' % type(self).__name__)
    def __delitem__(self, index):
        raise ValueError('%s may not be modified' % type(self).__name__)
    def __delslice__(self, i, j):
        raise ValueError('%s may not be modified' % type(self).__name__)
    def __iadd__(self, other):
        raise ValueError('%s may not be modified' % type(self).__name__)
    def __imul__(self, other):
        raise ValueError('%s may not be modified' % type(self).__name__)
    def append(self, v):
        raise ValueError('%s may not be modified' % type(self).__name__)
    def extend(self, v):
        raise ValueError('%s may not be modified' % type(self).__name__)
    def pop(self, v=None):
        raise ValueError('%s may not be modified' % type(self).__name__)
    def remove(self, v):
        raise ValueError('%s may not be modified' % type(self).__name__)
    def reverse(self):
        raise ValueError('%s may not be modified' % type(self).__name__)
    def sort(self):
        raise ValueError('%s may not be modified' % type(self).__name__)
    def __hash__(self):
        return self._hash

    def set_label(self, value):
        """
        Set the node label.  This will only succeed the first time the
        node label is set, which should occur in ImmutableTree.__init__().
        """
        if hasattr(self, '_label'):
            raise ValueError('%s may not be modified' % type(self).__name__)
        self._label = value

'''
class Rule(object):

    def __init__(self, lhs, rhs):
        # Represents the rule 'lhs -> rhs', where lhs is a non-terminal and
        # rhs is a list of non-terminals and terminals.
        self.lhs, self.rhs = lhs, rhs

    def __contains__(self, sym):
        return sym in self.rhs

    def __eq__(self, other):
        if type(other) is Rule:
            return self.lhs == other.lhs and self.rhs == other.rhs

        return False

    def __getitem__(self, i):
        return self.rhs[i]

    def __len__(self):
        return len(self.rhs)

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return self.lhs + ' -> ' + ' '.join(self.rhs)


class Grammar(object):
    """
    Represents a CFG.
    """

    def __init__(self):
        # The rules are represented as a dictionary from L.H.S to R.H.S.
        self.rules = defaultdict(list)

    def add(self, rule):
        """
        Adds the given rule to the grammar.
        """

        self.rules[rule.lhs].append(rule)

    @staticmethod
    def load_grammar(fpath):
        """
        Loads the grammar from file (from the )
        """

        grammar = Grammar()

        with open(fpath) as f:
            for line in f:
                line = line.strip()

                if len(line) == 0:
                    continue

                entries = line.split('->')
                lhs = entries[0].strip()
                for rhs in entries[1].split('|'):
                    grammar.add(Rule(lhs, rhs.strip().split()))

        return grammar

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        s = [str(r) for r in self.rules['S']]

        for nt, rule_list in self.rules.iteritems():
            if nt == 'S':
                continue

            s += [str(r) for r in rule_list]

        return '\n'.join(s)

    # Returns the rules for a given Non-terminal.
    def __getitem__(self, nt):
        return self.rules[nt]

    def is_terminal(self, sym):
        """
        Checks is the given symbol is terminal.
        """

        return len(self.rules[sym]) == 0

    def is_tag(self, sym):
        """
        Checks whether the given symbol is a tag, i.e. a non-terminal with rules
        to solely terminals.
        """

        if not self.is_terminal(sym):
            return all(self.is_terminal(s) for r in self.rules[sym] for s in
                       r.rhs)

        return False


class EarleyState(object):
    """
    Represents a state in the Earley algorithm.
    """

    GAM = '<GAM>'

    def __init__(self, rule, dot=0, sent_pos=0, chart_pos=0, back_pointers=[]):
        # CFG Rule.
        self.rule = rule
        # Dot position in the rule.
        self.dot = dot
        # Sentence position.
        self.sent_pos = sent_pos
        # Chart index.
        self.chart_pos = chart_pos
        # Pointers to child states (if the given state was generated using
        # Completer).
        self.back_pointers = back_pointers

    def __eq__(self, other):
        if type(other) is EarleyState:
            return self.rule == other.rule and self.dot == other.dot and \
                   self.sent_pos == other.sent_pos

        return False

    def __len__(self):
        return len(self.rule)

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        def str_helper(state):
            return ('(' + state.rule.lhs + ' -> ' +
                    ' '.join(state.rule.rhs[:state.dot] + ['*'] +
                             state.rule.rhs[state.dot:]) +
                    (', [%d, %d])' % (state.sent_pos, state.chart_pos)))

        return (str_helper(self) +
                ' (' + ', '.join(str_helper(s) for s in self.back_pointers) + ')')

    def next(self):
        """
        Return next symbol to parse, i.e. the one after the dot
        """

        if self.dot < len(self):
            return self.rule[self.dot]

    def is_complete(self):
        """
        Checks whether the given state is complete.
        """

        return len(self) == self.dot

    @staticmethod
    def init():
        """
        Returns the state used to initialize the chart in the Earley algorithm.
        """

        return EarleyState(Rule(EarleyState.GAM, ['S']))


class ChartEntry(object):
    """
    Represents an entry in the chart used by the Earley algorithm.
    """

    def __init__(self, states):
        # List of Earley states.
        self.states = states

    def __iter__(self):
        return iter(self.states)

    def __len__(self):
        return len(self.states)

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return '\n'.join(str(s) for s in self.states)

    def add(self, state):
        """
        Add the given state (if it hasn't already been added).
        """

        if state not in self.states:
            self.states.append(state)


class Chart(object):
    """
    Represents the chart used in the Earley algorithm.
    """

    def __init__(self, entries):
        # List of chart entries.
        self.entries = entries

    def __getitem__(self, i):
        return self.entries[i]

    def __len__(self):
        return len(self.entries)

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return '\n\n'.join([("Chart[%d]:\n" % i) + str(entry) for i, entry in
                            enumerate(self.entries)])

    @staticmethod
    def init(l):
        """
        Initializes a chart with l entries (Including the dummy start state).
        """

        return Chart([(ChartEntry([]) if i > 0 else
                       ChartEntry([EarleyState.init()])) for i in range(l)])


class EarleyParse(object):
    """
    Represents the Earley-generated parse for a given sentence according to a
    given grammar.
    """

    def __init__(self, sentence, grammar):
        self.words = sentence.split()
        self.grammar = grammar

        self.chart = Chart.init(len(self.words) + 1)

    def predictor(self, state, pos):
        """
        Earley Predictor.
        """

        for rule in self.grammar[state.next()]:
            self.chart[pos].add(EarleyState(rule, dot=0,
                                            sent_pos=state.chart_pos, chart_pos=state.chart_pos))

    def scanner(self, state, pos):
        """
        Earley Scanner.
        """

        if state.chart_pos < len(self.words):
            word = self.words[state.chart_pos]

            if any((word in r) for r in self.grammar[state.next()]):
                self.chart[pos + 1].add(EarleyState(Rule(state.next(), [word]),
                                                    dot=1, sent_pos=state.chart_pos,
                                                    chart_pos=(state.chart_pos + 1)))

    def completer(self, state, pos):
        """
        Earley Completer.
        """

        for prev_state in self.chart[state.sent_pos]:
            if prev_state.next() == state.rule.lhs:
                self.chart[pos].add(EarleyState(prev_state.rule,
                                                dot=(prev_state.dot + 1), sent_pos=prev_state.sent_pos,
                                                chart_pos=pos,
                                                back_pointers=(prev_state.back_pointers + [state])))

    def parse(self):
        """
        Parses the sentence by running the Earley algorithm and filling out the
        chart.
        """

        # Checks whether the next symbol for the given state is a tag.
        def is_tag(state):
            return self.grammar.is_tag(state.next())

        for i in range(len(self.chart)):
            for state in self.chart[i]:
                if not state.is_complete():
                    if is_tag(state):
                        self.scanner(state, i)
                    else:
                        self.predictor(state, i)
                else:
                    self.completer(state, i)

    def has_parse(self):
        """
        Checks whether the sentence has a parse.
        """

        for state in self.chart[-1]:
            if state.is_complete() and state.rule.lhs == 'S' and \
                    state.sent_pos == 0 and state.chart_pos == len(self.words):
                return True

        return False

    def get(self):
        """
        Returns the parse if it exists, otherwise returns None.
        """

        def get_helper(state):
            if self.grammar.is_tag(state.rule.lhs):
                return Tree(state.rule.lhs, [state.rule.rhs[0]])

            return Tree(state.rule.lhs,
                        [get_helper(s) for s in state.back_pointers])

        for state in self.chart[-1]:
            if state.is_complete() and state.rule.lhs == 'S' and \
                    state.sent_pos == 0 and state.chart_pos == len(self.words):
                return get_helper(state)

        return None


def main():
    grammar_file = 'sample-grammer.txt'

    grammar = Grammar.load_grammar(grammar_file)

    def run_parse(sentence):
        parse = EarleyParse(sentence, grammar)
        parse.parse()
        return parse.get()

    while True:
        try:
            sentence = input()

            # Strip the sentence of any puncutation.
            stripped_sentence = sentence
            for p in string.punctuation:
                stripped_sentence = stripped_sentence.replace(p, '')

            parse = run_parse(stripped_sentence)
            if parse is None:
                print(sentence + '\n')
            else:
                    parse.pretty_print()
        except EOFError:
            sys.exit()


if __name__ == '__main__':
    main()