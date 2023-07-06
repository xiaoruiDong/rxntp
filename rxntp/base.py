#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
This module provides the base class of reaction template.
"""

from copy import deepcopy
from typing import Callable, Optional, Union

import matplotlib.pyplot as plt
import networkx as nx

from rdkit.Chem import GetShortestPath
from rdmc import RDKitMol
from rdmc.reaction import Reaction
from rdmc.utils import PERIODIC_TABLE, CPK_COLOR_PALETTE


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

    def __neg__(self):
        # Perform negation of the graph
        return self.get_reverse()

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

    def draw(self,
             ax: Optional['Axes'] = None,
             layout_func: Callable = nx.spring_layout,
             ):
        """
        Draw the reaction template.
        """
        self._draw(self.graph, ax=ax, layout_func=layout_func)

    @staticmethod
    def _draw(graph,
              ax: Optional['Axes'] = None,
              layout_func: Callable = nx.spring_layout):
        """
        A helper function to draw the reaction template according to the provided graph.
        """
        if ax is None:
            _, ax = plt.subplots(1, 1)

        pos = layout_func(graph)

        # Plot nodes/atoms

        node_color = [CPK_COLOR_PALETTE[PERIODIC_TABLE.GetElementSymbol(atomic_num)]
                      for atomic_num in nx.get_node_attributes(graph, 'atomic_num').values()]
        nx.draw_networkx_nodes(graph,
                               pos,
                               node_color=node_color,
                               edgecolors='k',
                               node_size=1000,
                               ax=ax)

        # Plot edges/bonds
        style_book = \
            {'FORM_BOND': ('red', 'dashed', 3),
             'BREAK_BOND': ('green', 'dotted', 3),
             'CHANGE_BOND': ('black', 'solid', 3),
             'ORIG_BOND': ('grey', 'solid', 1),
             }
        edge_color, edge_style, edge_width = zip(*[style_book[bond_change_type]
                                                   for bond_change_type in nx.get_edge_attributes(graph,
                                                                                                  'bond_change_type').values()])
        edge_style = list(edge_style)  # Somehow in matplotlib, if edge_style is a tuple, it will run into bugs
        nx.draw_networkx_edges(graph,
                               pos,
                               edge_color=edge_color,
                               style=edge_style,
                               width=edge_width,
                               ax=ax)

        # Plot label information
        node_labels = {atom_idx: ReactionTemplate.get_node_status(atom_idx, attr_dict)
                       for atom_idx, attr_dict in graph.nodes(data=True)}
        nx.draw_networkx_labels(graph,
                                pos,
                                labels=node_labels,
                                font_size=7,
                                ax=ax,
                                )
        # Plot edge label information
        edge_labels = {(u,v): ReactionTemplate.get_edge_status(attr_dict, include_reactant_bond_order=False)
                       for u, v, attr_dict in graph.edges(data=True)}
        nx.draw_networkx_edge_labels(graph,
                                     pos,
                                     edge_labels=edge_labels,
                                     label_pos=0.5,
                                     font_size=7,
                                     ax=ax,
                                     rotate=False,)
        plt.show()

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
        return cls(graph,
                   num_reactants=reaction.num_reactants,
                   num_products=reaction.num_products)

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
            return Reaction(reactant=reaction.product,
                            product=reaction.reactant,)
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
        for atom_idx in sorted(involved_atoms):
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




    @staticmethod
    def get_node_status(atom_idx: int,
                        attr_dict: dict):
        """
        Generate the node status description for visualization and export.

        Args:
            atom_idx (int): The atom index
            attr_dict (dict): The attribute dictionary of the node.

        Returns:
            str: the status of the atom
        """
        atom_symbol = PERIODIC_TABLE.GetElementSymbol(attr_dict['atomic_num'])
        status = f'{atom_symbol}:{atom_idx+1}\n'

        if attr_dict['radical_change'] > 0:
            status += 'GAIN RADICAL\n'
        elif attr_dict['radical_change'] < 0:
            status += 'LOSE RADICAL\n'
        if attr_dict['charge_change'] > 0:
            status += 'GAIN CHARGE\n'
        elif attr_dict['charge_change'] < 0:
            status += 'LOSE CHARGE\n'
        if attr_dict['lone_pair_change'] > 0:
            status += 'GAIN PAIR\n'
        elif attr_dict['lone_pair_change'] < 0:
            status += 'LOSE PAIR\n'
        if status.endswith('\n'):
            status = status[:-1]

        return status

    @staticmethod
    def get_edge_status(attr_dict: dict,
                        include_reactant_bond_order: bool = False):
        """
        Generate the edge status description for visualization and export.

        Args:
            attr_dict (dict): The attribute dictionary of the edge.
            include_reactant_bond_order (bool): How important is th reactant bond order is still under investigation.
                                                Here, we make it available so that it can be easily turned on/off.

        Returns:
            str: the status of the edge
        """
        bond_change_type = attr_dict['bond_change_type']
        if bond_change_type in ["FORM_BOND", "BREAK_BOND"]:
            # Example: "FORM_BOND: HOMO"; "BREAK_BOND: HETERO"
            status = f"{bond_change_type}: {'HOMO' if attr_dict['homo'] else 'HETERO'}"
        elif bond_change_type == 'CHANGE_BOND':
            status = f"{bond_change_type}: {attr_dict['bond_order_change']:+.1f}"
        else:  # ORIG_BOND
            status = f"{bond_change_type}: {attr_dict['reactant_bond_order']:.1f}"

        if include_reactant_bond_order:
            status += f"REACTANT BO: {attr_dict['reactant_bond_order']:.1f}"

        return status

    def get_reverse(self,
                    name: Optional[str] = None,
                    ) -> "ReactionTemplate":
        """
        Generate the reverse reaction's template
        """
        reverse_template = self.copy()
        r_graph = reverse_template.graph

        for node in r_graph.nodes:
            node_attr = r_graph.nodes[node]
            node_attr['radical_change'] = - node_attr['radical_change']
            node_attr['charge_change'] = - node_attr['charge_change']
            node_attr['lone_pair_change'] = - node_attr['lone_pair_change']

        for edge in r_graph.edges:
            edge_attr = r_graph.edges[edge]
            if edge_attr['bond_change_type'] == 'FORM_BOND':
                edge_attr['bond_change_type'] = 'BREAK_BOND'
            elif edge_attr['bond_change_type'] == 'BREAK_BOND':
                edge_attr['bond_change_type'] = 'FORM_BOND'
            if edge_attr.get('reactant_bond_order', None) is not None:
                edge_attr['reactant_bond_order'] = edge_attr['reactant_bond_order'] + edge_attr['bond_order_change']
            edge_attr['bond_order_change'] = - edge_attr['bond_order_change']

        if name is not None:
            reverse_template.name = name
        elif self.name.endswith('_rev'):
            reverse_template.name = self.name[:-4]
        else:
            reverse_template.name = self.name + '_rev'
        reverse_template.num_reactants, reverse_template.num_products = reverse_template.num_products, reverse_template.num_reactants

        return reverse_template

    def renumber(self,
                 mapping: Optional[dict] = None,
                 copy: bool = True,):
        """
        Renumber the nodes in the graph by a mapping.
        """
        if copy:
            new_template = self.copy()
        else:
            new_template = self
        if mapping is None:
            mapping = {node: i for i, node in enumerate(self.graph.nodes)}
        new_template.graph = nx.relabel_nodes(self.graph, mapping, copy=True)
        return new_template
