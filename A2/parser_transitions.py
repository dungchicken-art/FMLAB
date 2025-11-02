#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys

class PartialParse(object):
    def __init__(self, sentence):
        self.sentence = sentence[:]      
        self.stack = ["ROOT"]
        self.buffer = sentence[:]
        self.dependencies = []

    def parse_step(self, transition):
        if transition == "S":
            if len(self.buffer) > 0:
                self.stack.append(self.buffer.pop(0))
        elif transition == "LA":
            head = self.stack[-1]
            dependent = self.stack[-2]
            self.dependencies.append((head, dependent))
            del self.stack[-2]
        elif transition == "RA":
            head = self.stack[-2]
            dependent = self.stack[-1]
            self.dependencies.append((head, dependent))
            self.stack.pop()
        else:
            raise ValueError(f"Unknown transition: {transition}")

    def parse(self, transitions):
        for transition in transitions:
            self.parse_step(transition)
        return self.dependencies


def minibatch_parse(sentences, model, batch_size):
    partial_parses = []
    for sentence in sentences:
        pp = PartialParse(sentence)
        pp.sentence = sentence  
        partial_parses.append(pp)

    unfinished_parses = partial_parses[:]
    while len(unfinished_parses) > 0:
        batch = unfinished_parses[:batch_size]
        transitions = model.predict(batch)
        for i, transition in enumerate(transitions):
            batch[i].parse_step(transition)
        unfinished_parses = [pp for pp in unfinished_parses if not (len(pp.buffer) == 0 and len(pp.stack) == 1)]
    dependencies = [pp.dependencies for pp in partial_parses]
    return dependencies


def test_step(name, transition, stack, buf, deps, ex_stack, ex_buf, ex_deps):
    pp = PartialParse([])
    pp.stack, pp.buffer, pp.dependencies = stack, buf, deps
    pp.parse_step(transition)
    stack, buf, deps = (tuple(pp.stack), tuple(pp.buffer), tuple(sorted(pp.dependencies)))
    assert stack == ex_stack
    assert buf == ex_buf
    assert deps == ex_deps
    print(f"{name} test passed!")


def test_parse_step():
    test_step("SHIFT", "S", ["ROOT", "the"], ["cat", "sat"], [],
              ("ROOT", "the", "cat"), ("sat",), ())
    test_step("LEFT-ARC", "LA", ["ROOT", "the", "cat"], ["sat"], [],
              ("ROOT", "cat"), ("sat",), (("cat", "the"),))
    test_step("RIGHT-ARC", "RA", ["ROOT", "run", "fast"], [], [],
              ("ROOT", "run"), (), (("run", "fast"),))


def test_parse():
    sentence = ["parse", "this", "sentence"]
    dependencies = PartialParse(sentence).parse(["S", "S", "S", "LA", "RA", "RA"])
    dependencies = tuple(sorted(dependencies))
    expected = (('ROOT', 'parse'), ('parse', 'sentence'), ('sentence', 'this'))
    assert dependencies == expected
    assert tuple(sentence) == ("parse", "this", "sentence")
    print("parse test passed!")


class DummyModel(object):
    def __init__(self, mode="unidirectional"):
        self.mode = mode

    def predict(self, partial_parses):
        if self.mode == "unidirectional":
            return self.unidirectional_predict(partial_parses)
        elif self.mode == "interleave":
            return self.interleave_predict(partial_parses)
        else:
            raise NotImplementedError()

    def unidirectional_predict(self, partial_parses):
        return [("RA" if pp.stack[1] == "right" else "LA") if len(pp.buffer) == 0 else "S"
                for pp in partial_parses]

    def interleave_predict(self, partial_parses):
        return [("RA" if len(pp.stack) % 2 == 0 else "LA") if len(pp.buffer) == 0 else "S"
                for pp in partial_parses]


def test_dependencies(name, deps, ex_deps):
    deps = tuple(sorted(deps))
    assert deps == ex_deps


def test_minibatch_parse():
    sentences = [["right", "arcs", "only"],
                 ["right", "arcs", "only", "again"],
                 ["left", "arcs", "only"],
                 ["left", "arcs", "only", "again"]]
    deps = minibatch_parse(sentences, DummyModel(), 2)
    test_dependencies("minibatch_parse", deps[0],
                      (('ROOT', 'right'), ('arcs', 'only'), ('right', 'arcs')))
    test_dependencies("minibatch_parse", deps[1],
                      (('ROOT', 'right'), ('arcs', 'only'), ('only', 'again'), ('right', 'arcs')))
    test_dependencies("minibatch_parse", deps[2],
                      (('only', 'ROOT'), ('only', 'arcs'), ('only', 'left')))
    test_dependencies("minibatch_parse", deps[3],
                      (('again', 'ROOT'), ('again', 'arcs'), ('again', 'left'), ('again', 'only')))
    sentences = [["right"]]
    deps = minibatch_parse(sentences, DummyModel(), 2)
    test_dependencies("minibatch_parse", deps[0], (('ROOT', 'right'),))
    sentences = [["this", "is", "interleaving", "dependency", "test"]]
    deps = minibatch_parse(sentences, DummyModel(mode="interleave"), 1)
    test_dependencies("minibatch_parse", deps[0],
                      (('ROOT', 'is'), ('dependency', 'interleaving'),
                      ('dependency', 'test'), ('is', 'dependency'), ('is', 'this')))
    print("minibatch_parse test passed!")


if __name__ == '__main__':
    args = sys.argv
    if len(args) != 2:
        raise Exception("You did not provide a valid keyword. Either provide 'part_c' or 'part_d'")
    elif args[1] == "part_c":
        test_parse_step()
        test_parse()
    elif args[1] == "part_d":
        test_minibatch_parse()
    else:
        raise Exception("You did not provide a valid keyword. Either provide 'part_c' or 'part_d'")
