#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
This module provides the base class of reaction template.
"""
import networkx as nx

class ReactionTemplate(object):

    def __init__(self,
                 graph: nx.Graph,
                 num_reactants: int = 1,
                 num_products: int = 1,
                 name: str = '',
                 reversible: bool = True,
                 ):
        """
        """
        self.graph = graph
        self.name = name
        self.reversible = reversible
        self.num_reactants = num_reactants
        self.num_products = num_products

    @property
    def num_atoms(self):
        return self.graph.number_of_nodes()

    @property
    def num_bonds(self):
        return self.graph.number_of_edges()

    @property
    def formed_bonds(self):
        return [tuple(sorted((u, v))) for u, v, d in self.graph.edges(data='bond_change_type') if d == 'FORM_BOND']

    @property
    def broken_bonds(self):
        return [tuple(sorted((u, v))) for u, v, d in self.graph.edges(data='bond_change_type') if d == 'BREAK_BOND']

    @property
    def non_react_bonds(self):
        return [tuple(sorted((u, v))) for u, v, d in self.graph.edges(data='bond_order_change') if d == 0]

    @property
    def changed_bonds(self):
        return [tuple(sorted((u, v))) for u, v, d in self.graph.edges(data='bond_change_type') if d == 'CHANGE_BOND']

    @property
    def is_charge_balance(self):
        return sum([d['charge_change'] for _, d in self.graph.nodes(data=True)]) == 0

    @property
    def is_mult_equal(self):
        return sum([d['radical_change'] for _, d in self.graph.nodes(data=True)]) == 0

    @property
    def num_formed_bonds(self):
        return len(self.formed_bonds)

    @property
    def num_broken_bonds(self):
        return len(self.broken_bonds)

    @property
    def num_non_react_bonds(self):
        return len(self.non_react_bonds)

    @property
    def num_changed_bonds(self):
        return len(self.changed_bonds)

    def copy(self):
        """
        Make a copy of the reaction template.
        """
        return ReactionTemplate(deepcopy(self.graph),
                                num_reactants=self.num_reactants,
                                num_products=self.num_products,
                                name=self.name,
                                reversible=self.reversible)

