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

