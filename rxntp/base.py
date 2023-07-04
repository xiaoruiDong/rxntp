#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
This module provides the base class of reaction template.
"""

from copy import deepcopy
from typing import Callable, Optional, Union
import networkx as nx
from rdmc import RDKitMol
from rdmc.reaction import Reaction

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

    @classmethod
    def from_reaction(cls,
                      reaction: Reaction,
                      ) -> 'ReactionTemplate' :
        """
        Generate a reaction template from a reaction.

        Args:
            reaction (Reaction): An RDMC reaction object.
        """
        graph = cls.make_graph_from_reaction(reaction)
        return cls(graph)

    @classmethod
    def from_reaction_smiles(cls, rxn_smiles: str):
        """
        Generate a reaction template from a reaction SMILES.

        Args:
            rxn_smiles (str): the SMILES for the reaction.
        """
        rxn = Reaction.from_reaction_smiles(rxn_smiles)
        return cls.from_reaction(rxn)

    @classmethod
    def from_reactant_and_product_smiles(cls, rsmi: str, psmi: str):
        """
        Generate a reaction template from reactant and product SMILES.

        Args:
            rsmi (str): the SMILES for the reactant.
            psmi (str): the SMILES for the product.
        """
        rxn = Reaction.from_reactant_and_product_smiles(rsmi=rsmi, psmi=psmi)
        return cls.from_reaction(rxn)

    @staticmethod
    def correct_reaction_direction(reaction):
        """
        Correct the reaction direction. If more molecules as the reactants than as the products,
        the reaction is reversed; if more bonds are formed than broken, the reaction is reversed.

        Args:
            reaction (Reaction): the reaction to be corrected.

        Returns:
            reaction (Reaction): the corrected reaction.
        """
        reverse_flag = False
        # Case 1: more formed bonds than broken bonds
        if reaction.num_reactants == reaction.num_products:
            formed_bonds, broken_bonds = reaction.formed_bonds, reaction.broken_bonds
            if formed_bonds > broken_bonds:
                reverse_flag = True
        # Case 2: more molecules as the reactants than as the products
        elif reaction.num_reactants > reaction.num_products:
            reverse_flag = True
        if reverse_flag:
            return reaction.apply_reaction_direction_correction()
        return reaction

    @staticmethod
    def make_graph_from_reaction(reaction: Reaction,
                                 correct_resonance: bool = True,
                                 correct_reaction_direction: bool = True,
                                ):
        """
        Make a graph for the reaction template.
        """
        if correct_resonance:
            reaction = reaction.apply_resonance_correction()

        if correct_reaction_direction:
            reaction = ReactionTemplate.correct_reaction_direction(reaction)

        involved_atoms = reaction.involved_atoms
        involved_bonds = reaction.involved_bonds
        rmol, pmol = reaction.reactant_complex, reaction.product_complex

        graph = nx.Graph()

        # Add nodes
        for atom_idx in involved_atoms:
            ratom, patom = rmol.GetAtomWithIdx(atom_idx), pmol.GetAtomWithIdx(atom_idx)
            attr_dict = ReactionTemplate._get_node_attr_from_atom(ratom, patom)
            graph.add_node(**attr_dict)

        # Add edges that are changed in the reaction
        for bond_idxs in involved_bonds:
            attr_dict = ReactionTemplate._get_edge_attr_from_bond(rmol, pmol, bond_idxs)
            if attr_dict is not None:
                graph.add_edge(**attr_dict)

        # Expand the graph based on original connectivity
        for i in range(len(involved_atoms)):
            for j in range(i, len(involved_atoms)):
                atom1, atom2 = involved_atoms[i], involved_atoms[j]
                if (atom1, atom2) in involved_bonds or (atom2, atom1) in involved_bonds:
                    continue
                if rmol.GetBondBetweenAtoms(atom1, atom2):
                    attr_dict = ReactionTemplate._get_edge_attr_from_bond(rmol, pmol, (atom1, atom2))
                    graph.add_edge(**attr_dict)

        return graph

    @staticmethod
    def _get_node_attr_from_atom(ratom: 'Atom',
                                 patom: 'Atom',
                                 ) -> dict:
        """
        Generate a node's attribute for the reaction graph from the atom in reactant and product.

        Args:
            ratom (Atom): the atom in reactant.
            patom (Atom): the atom in product.

        Returns:
            attr_dict (dict): the attribute dictionary for the node.
        """
        atom_idx = ratom.GetIdx()
        atomic_num = ratom.GetAtomicNum()

        # Calculate lone pair changes
        radical_change = patom.GetNumRadicalElectrons() - ratom.GetNumRadicalElectrons()
        charge_change = patom.GetFormalCharge() - ratom.GetFormalCharge()
        valence_change = patom.GetTotalValence() - ratom.GetTotalValence()
        # Left the formula below for reference
        # r_lone_pair = n_out_electrons - ratom.GetNumRadicalElectrons() - \
        #                 ratom.GetFormalCharge() - ratom.GetTotalValence()
        # p_lone_pair = n_out_electrons - patom.GetNumRadicalElectrons() - \
        #                 patom.GetFormalCharge() - patom.GetTotalValence()
        # lone_pair_change = p_lone_pair - r_lone_pair
        lone_pair_change = - radical_change - charge_change - valence_change

        attr_dict = {
            'node_for_adding': atom_idx,
            'atomic_num': atomic_num,
            'radical_change': radical_change,
            'charge_change': charge_change,
            'lone_pair_change': lone_pair_change,
        }

        return attr_dict

    @staticmethod
    def _determine_bond_change(rmol: RDKitMol,
                               pmol: RDKitMol,
                               bond_idxs: tuple,
                               ) -> tuple:
        """
        Determine the bond change type.

        Args:
            rmol (RDKitMol): the reactant complex.
            pmol (RDKitMol): the product complex.
            bond_idxs (tuple): the indices of the bond.

        returns:
            tuple:
            - bond_change_type (str): the type of bond change.
            - homo (bool): if the bond change is homogeneous or heterogeneous.
            - bond_order_change (float): the bond order change.
            - r_order (float): the bond order in the reactant.
        """
        r_bond = rmol.GetBondBetweenAtoms(*bond_idxs)
        p_bond = pmol.GetBondBetweenAtoms(*bond_idxs)
        r_order = 0 if r_bond is None else r_bond.GetBondTypeAsDouble()
        p_order = 0 if p_bond is None else p_bond.GetBondTypeAsDouble()
        bond_order_change = p_order - r_order

        if r_order and p_order:
            homo = True
            if bond_order_change:
                bond_change_type = 'CHANGE_BOND'
            else:
                bond_change_type = 'ORIG_BOND'
        elif r_order and not p_order:
            homo = GetShortestPath(pmol._mol, *bond_idxs) != ()
            bond_change_type = 'BREAK_BOND'
        elif p_order and not r_order:
            homo = GetShortestPath(rmol._mol, *bond_idxs) != ()
            bond_change_type = 'FORM_BOND'
        else:
            homo = None  # Currently, NO_BOND is not stored and would not cause a problem later
            bond_change_type = 'NO_BOND'
        return (bond_change_type, homo, bond_order_change, r_order)

    @ staticmethod
    def _get_edge_attr_from_bond(rmol: RDKitMol,
                                 pmol: RDKitMol,
                                 bond_idxs: tuple,
                                 ):
        """
        Get an edge's attribute from the bond of the reactant and the product.
        """
        bond_change_type, homo, bond_order_change, r_order \
            = ReactionTemplate._determine_bond_change(rmol, pmol, bond_idxs)

        if bond_change_type != 'NO_BOND':
            attr_dict = {
                'u_of_edge': bond_idxs[0],
                'v_of_edge': bond_idxs[1],
                'bond_change_type': bond_change_type,
                'bond_order_change': bond_order_change,
                'reactant_bond_order': r_order,
                'homo': homo,
            }
            return attr_dict

    def add_node_from_atom(self,
                           ratom: 'Atom',
                           patom: 'Atom',
                           ) -> None:
        """
        Add a node to the graph from the atom in reactant and product.

        Args:
            ratom (Atom): the atom in reactant.
            patom (Atom): the atom in product.
        """
        attr_dict = self._get_node_attr_from_atom(ratom, patom)
        self.graph.add_node(**attr_dict)

    def add_edge_from_bond(self,
                           rmol: RDKitMol,
                           pmol: RDKitMol,
                           bond_idxs: tuple,):
        """
        """
        attr_dict = self._get_edge_attr_from_bond(rmol, pmol, bond_idxs)
        if attr_dict is not None:
            self.graph.add_edge(**attr_dict)



