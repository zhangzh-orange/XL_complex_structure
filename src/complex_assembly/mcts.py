#! python3
# -*- encoding: utf-8 -*-
# ===================================
# File    :   mcts.py
# Time    :   2025/12/08 12:27:15
# Author  :   Zehong Zhang
# Contact :   zhang@fmp-berlin.de
# ===================================

import argparse
import sys
import os
import numpy as np
import pandas as pd
import glob
import itertools
import math
import copy
from collections import Counter, defaultdict
from Bio.SVDSuperimposer import SVDSuperimposer
import pdb
from utils import parse_atm_record, read_cif

# TODO: 1. Read inputs:
# parser = argparse.ArgumentParser(description = '''Find optimal paths by Monte Carlo Tree Search.''')
# parser.add_argument('--network', nargs=1, type= str, default=sys.stdin, help = 'Path to csv containing pairwise interactions.')
# parser.add_argument('--pairdir', nargs=1, type= str, default=sys.stdin, help = 'Path to dir containing all connecting pairs')
# parser.add_argument('--plddt_dir', nargs=1, type= str, default=sys.stdin, help = 'Path to dir containing plDDT scores for each complex')
# parser.add_argument('--useqs', nargs=1, type= str, default=sys.stdin, help = 'CSV with unique seqs')
# parser.add_argument('--chain_seqs', nargs=1, type= str, default=sys.stdin, help = 'CSV with mapping btw useqs and chains')
# parser.add_argument('--outdir', nargs=1, type= str, default=sys.stdin, help = 'Where to write all complexes')

##############FUNCTIONS##############




def check_overlaps(cpath_coords,cpath_CA_inds,n_coords,n_CA_inds):
    '''Check assembly overlaps of new chain

    Parameters
    ----------
    cpath_coords : list of np.ndarray
        Coordinates of all atoms in currently assembled chains.
        Each element corresponds to one chain: [chain1_coords, chain2_coords, ...]
        Shape per chain: (n_atoms_i, 3) where 3 represents (x, y, z) coordinates
    
    cpath_CA_inds : list of np.ndarray
        Indices of alpha-carbon (Cα) atoms for each assembled chain.
        Used to extract backbone atoms from full coordinate sets.
        Example: [[0, 1, 4, ...], [0, 1, 4, ...], ...] where indices point to Cα positions
    
    n_coords : np.ndarray
        Coordinates of all atoms in the new chain to be added.
        Shape: (n_atoms_new, 3) - all atom positions for the candidate chain
    
    n_CA_inds : np.ndarray
        Indices of alpha-carbon (Cα) atoms in the new chain.
        Used to extract backbone from the new chain's full atom set.
        Example: [0, 1, 4, 5, ...] indices corresponding to Cα atoms
    '''
    #Check CA overlap
    n_CAs = n_coords[n_CA_inds] #New chain CAs
    l1 = (len(n_CAs))
    #Go through all previous CAs and compare

    for i in range(len(cpath_coords)):
        p_CAs = cpath_coords[i][cpath_CA_inds[i]]
        #Calc 2-norm
        mat = np.append(n_CAs, p_CAs,axis=0)
        a_min_b = mat[:,np.newaxis,:] -mat[np.newaxis,:,:]
        dists = np.sqrt(np.sum(a_min_b.T ** 2, axis=0)).T
        contact_dists = dists[l1:,:l1]
        overlap = np.argwhere(contact_dists<=5) #5 Å threshold
        if overlap.shape[0]>0.5*min(l1,len(p_CAs)): #If over 50% overlap
            return True

    return False


def score_interfaces(path_coords, path_CB_inds, path_plddt):
    '''Score all interfaces in the current complex
    '''

    interfaces_score = 0
    chain_inds = np.arange(len(path_coords))
    #Get interfaces per chain
    for i in chain_inds:
        chain_coords = path_coords[i]
        chain_CB_inds = path_CB_inds[i]
        l1 = len(chain_CB_inds)
        chain_CB_coords = chain_coords[chain_CB_inds]
        chain_plddt = path_plddt[i]
        
        # 获取除当前链i外的所有其他链索引
        other_chains = np.setdiff1d(chain_inds, i)
        for int_i in other_chains:
            int_chain_CB_coords = path_coords[int_i][path_CB_inds[int_i]]
            int_chain_plddt = path_plddt[int_i]
            #Calc 2-norm
            mat = np.append(chain_CB_coords,int_chain_CB_coords,axis=0)
            a_min_b = mat[:,np.newaxis,:] -mat[np.newaxis,:,:]
            dists = np.sqrt(np.sum(a_min_b.T ** 2, axis=0)).T
            contact_dists = dists[:l1,l1:]
            contacts = np.argwhere(contact_dists<=8)
            #The first axis contains the contacts from chain 1
            #The second the contacts from chain 2
            if contacts.shape[0]>0:
                av_if_plDDT =  np.concatenate((chain_plddt[contacts[:,0]], int_chain_plddt[contacts[:,1]])).mean()
                interfaces_score +=  np.log10(contacts.shape[0]+1)*av_if_plDDT
    
    return interfaces_score


def read_crosslinks(csv_path):
    pass

def score_crosslinks(path_coords, path_CA_inds):
    # TODO: 计算输入complex中所有crosslink的满足度，crosslink满足数目越多，打分越高
    # 取CA算距离
    crosslinks = {
        "start_CA_chains":[],
        "start_CA_inds":[],
        "end_CA_chains":[],
        "end_CA_inds":[]
        }
    crosslinker_length = 35 # A 限制距离
    total_links = len(crosslinks["start_CA_inds"]) + 1e-6

    satisfied_links = 0
    for s,e in zip(crosslinks["start_CA_inds"],crosslinks["end_CA_inds"]):
        # 计算欧几里得距离
        # 定义两个三维点
        point1 = np.array([1, 2, 3])
        point2 = np.array([4, 6, 8])

        # 计算差值向量
        diff = point1 - point2

        # 计算欧几里得距离
        distance = np.sqrt(np.sum(diff**2))
        if distance <= crosslinker_length:
            satisfied_links += 1
    crosslinks_score = satisfied_links/total_links
    return crosslinks_score

def score_complex():
    # TODO: complex_score = interfaces_score * crosslinks_score
    # return complex_score
    pass


class MonteCarloTreeSearchNode():
    '''Based on https://ai-boson.github.io/mcts/
    Each node is a decision - not a chain. Before a new chain is added, the parents
    therefore have to be checked so that the new chain is not in the path.
    '''
    def __init__(self, chain, edge_chain, chain_coords, chain_CA_inds, chain_CB_inds, chain_pdb, chain_plddt,
                source=None, complex_scores=[0], parent=None, parent_path=[], total_chains=0):
        self.chain = chain
        self.edge_chain = edge_chain
        self.chain_coords = chain_coords
        self.CA_inds = chain_CA_inds
        self.CB_inds = chain_CB_inds
        self.pdb = chain_pdb
        self.plddt = chain_plddt
        self.source = source #where the chain comes from

        #These are the scores obtained from all rollouts from the current node
        #These are averaged to obtain a score for how well the current node performs in general
        self.complex_scores = complex_scores #sum over complex: (avg_if_plddt*log10(n_if_contacts))

        self.parent = parent #Parent node
        self.path = copy.deepcopy(parent_path) #All nodes up to (and including) the parent
        self.path.append(chain)
        self.children = [] #All nodes branching out from the current
        self._number_of_visits = 0
        self._untried_edges, self._untried_sources = self.get_possible_edges()
        self.total_chains=total_chains

        return

    def get_possible_edges(self):
        '''Get all possible edges in path
        '''

        untried_edges = []
        untried_sources = []
        for chain in self.path:
            #Get all edges to the current node
            cedges = edges[np.argwhere(edges==chain)[:,0]]
            csources = sources[np.argwhere(edges==chain)[:,0]]
            #Go through all edges and see that the new connection is not in path
            for i in range(len(cedges)):
                diff = np.setdiff1d(cedges[i],self.path)
                if len(diff)>0:
                    untried_edges.append(cedges[i])
                    untried_sources.append(csources[i])

        return untried_edges, untried_sources

    def expand(self):
        '''Expand the path by selecting a random action
        '''
        new_edge = self._untried_edges.pop()
        new_source = self._untried_sources.pop()
        new_chain = np.setdiff1d(new_edge, self.path)[0]
        edge_chain =  np.setdiff1d(new_edge, new_chain)[0]
        #Read the pdb file containing the new edge
        if os.path.exists(pairdir+new_source+'_'+new_edge[0]+'-'+new_source+'_'+new_edge[1]+'.pdb'):
            pdb_chains, chain_coords, chain_CA_inds, chain_CB_inds = read_pdb(pairdir+new_source+'_'+new_edge[0]+'-'+new_source+'_'+new_edge[1]+'.pdb')
        else:
            pdb_chains, chain_coords, chain_CA_inds, chain_CB_inds = read_pdb(pairdir+new_source+'_'+new_edge[1]+'-'+new_source+'_'+new_edge[0]+'.pdb')

        #plDDT - for scoring
        source_plDDT =  np.load(plddt_dir+new_source+'.npy')
        si = 0
        for p_chain in new_source.split('_')[-1]:
            if p_chain==new_chain:
                new_chain_plddt=source_plDDT[si:si+chain_lens[new_chain]]
            else:
                si += chain_lens[p_chain]

        #Align the overlapping chains
        #Get the coords for the other chain in the edge
        edge_node = self
        while edge_node.chain!=edge_chain:
            edge_node = edge_node.parent


        #Set the coordinates to be superimposed.
        #coords will be put on top of reference_coords.
        sup = SVDSuperimposer()
        sup.set(edge_node.chain_coords,np.array(chain_coords[edge_chain])) #(reference_coords, coords)
        sup.run()
        rot, tran = sup.get_rotran()

        #Rotate coords from new chain to its new relative position/orientation
        rotated_coords = np.dot(np.array(chain_coords[new_chain]), rot) + tran

        #Get all chain coords and CA inds for the current path
        path_coords = []
        path_CA_inds =[]
        path_CB_inds = []
        path_plddt = []
        path_node = self
        while path_node.parent:
            path_coords.append(path_node.chain_coords)
            path_CA_inds.append(path_node.CA_inds)
            path_CB_inds.append(path_node.CB_inds)
            path_plddt.append(path_node.plddt)
            path_node = path_node.parent
        path_coords.append(path_node.chain_coords) #Last - when no parent = root
        path_CA_inds.append(path_node.CA_inds)
        path_CB_inds.append(path_node.CB_inds)
        path_plddt.append(path_node.plddt)


        #Check overlaps
        overlap = check_overlaps(path_coords,path_CA_inds,rotated_coords,chain_CA_inds[new_chain])

        #If no overlap - score and create a child node
        if overlap==False:
            #Add the new chain
            path_coords.append(rotated_coords), path_CB_inds.append(chain_CB_inds[new_chain]), path_plddt.append(new_chain_plddt)
            complex_score = score_complex(path_coords, path_CB_inds, path_plddt)
            #Add to all complex scores
            child_node = MonteCarloTreeSearchNode(new_chain, edge_chain, rotated_coords, np.array(chain_CA_inds[new_chain]),
                    np.array(chain_CB_inds[new_chain]), np.array(pdb_chains[new_chain]), new_chain_plddt,
                    source=new_source, complex_scores=[complex_score], parent=self, parent_path=self.path, total_chains=self.total_chains)

            self.children.append(child_node)
            return child_node

        else:
            #Only consider nodes that do not result in overlaps
            return None





    def rollout(self):
        '''Simulate an assembly path until
        1. all chains are in complex
        2. an overlap is found
        '''
        overlap = False


        #Get all chain coords, CA,CB inds and plddt for the current path
        rollout_path = []
        path_coords = []
        path_CA_inds =[]
        path_CB_inds = []
        path_plddt = []
        path_node = self
        while path_node.parent:
            rollout_path.append(path_node.chain)
            path_coords.append(path_node.chain_coords)
            path_CA_inds.append(path_node.CA_inds)
            path_CB_inds.append(path_node.CB_inds)
            path_plddt.append(path_node.plddt)
            path_node = path_node.parent
        rollout_path.append(path_node.chain)
        path_coords.append(path_node.chain_coords) #Last - when no parent = root
        path_CA_inds.append(path_node.CA_inds)
        path_CB_inds.append(path_node.CB_inds)
        path_plddt.append(path_node.plddt)

        def get_possible_edges(path):
            '''Get all possible edges in path
            Can't pass self - which is why this has to be defined here
            '''

            untried_edges = []
            untried_sources = []
            for chain in path:
                #Get all edges to the current node
                cedges = edges[np.argwhere(edges==chain)[:,0]]
                csources = sources[np.argwhere(edges==chain)[:,0]]
                #Go through all edges and see that the new connection is not in path
                for i in range(len(cedges)):
                    diff = np.setdiff1d(cedges[i],path)
                    if len(diff)>0:
                        untried_edges.append(cedges[i])
                        untried_sources.append(csources[i])

            return untried_edges, untried_sources

        while len(rollout_path)<self.total_chains and overlap==False:

            #Get untried edges and sources
            untried_edges, untried_sources = get_possible_edges(rollout_path)
            #Pick a random edge
            if len(untried_edges)>0:
                edge_ind = np.random.randint(len(untried_edges))
            else:
                overlap=True
                break
            new_edge = untried_edges[edge_ind]
            new_source = untried_sources[edge_ind]
            new_chain = np.setdiff1d(new_edge, rollout_path)[0]
            edge_chain = np.setdiff1d(new_edge, new_chain)[0]

            #Read the pdb file containing the new edge
            if os.path.exists(pairdir+new_source+'_'+new_edge[0]+'-'+new_source+'_'+new_edge[1]+'.pdb'):
                pdb_chains, chain_coords, chain_CA_inds, chain_CB_inds = read_pdb(pairdir+new_source+'_'+new_edge[0]+'-'+new_source+'_'+new_edge[1]+'.pdb')
            else:
                pdb_chains, chain_coords, chain_CA_inds, chain_CB_inds = read_pdb(pairdir+new_source+'_'+new_edge[1]+'-'+new_source+'_'+new_edge[0]+'.pdb')

            #plDDT - for scoring
            source_plDDT =  np.load(plddt_dir+new_source+'.npy')
            si = 0
            for p_chain in new_source.split('_')[-1]:
                if p_chain==new_chain:
                    new_chain_plddt=source_plDDT[si:si+chain_lens[new_chain]]
                else:
                    si += chain_lens[p_chain]

            #Align the overlapping chains
            #Get the coords for the other chain in the edge
            edge_ind = np.argwhere(np.array(rollout_path)==edge_chain)[0][0]
            #Set the coordinates to be superimposed.
            #coords will be put on top of reference_coords.
            sup = SVDSuperimposer()
            sup.set(path_coords[edge_ind],np.array(chain_coords[edge_chain])) #(reference_coords, coords)
            sup.run()
            rot, tran = sup.get_rotran()

            #Rotate coords from new chain to its new relative position/orientation
            rotated_coords = np.dot(np.array(chain_coords[new_chain]), rot) + tran

            #Check overlaps
            overlap = check_overlaps(path_coords,path_CA_inds,rotated_coords,chain_CA_inds[new_chain])


            #If no overlap - score and create a child node
            if overlap==False:
                #Add the new chain
                rollout_path.append(new_chain), path_coords.append(rotated_coords),
                path_CA_inds.append(chain_CA_inds[new_chain])
                path_CB_inds.append(chain_CB_inds[new_chain]), path_plddt.append(new_chain_plddt)

            else:
                break
        #Score rollout
        rollout_score = score_complex(path_coords, path_CB_inds, path_plddt)

        return rollout_score


    def back_prop(self, rollout_score):
        '''Update the previous nodes in the path
        '''

        self._number_of_visits += 1
        self.complex_scores.append(rollout_score)
        #This is recursive and will back_prop to all parents
        if self.parent:
            self.parent.back_prop(rollout_score)

    def best_child(self):
        '''Calculate the UCB

        Vi is the average reward/value of all nodes beneath this node (sum of interface scores)
        N is the number of times the parent node has been visited, and
        ni is the number of times the child node i has been visited

        The first component of the formula above corresponds to exploitation;
        it is high for moves with high average win ratio.
        The second component corresponds to exploration; it is high for moves with few simulations.
        '''

        choices_weights = [(np.average(c.complex_scores) + 2 * np.sqrt(np.log(c.parent._number_of_visits+1e-3) / (c._number_of_visits+1e-12))) for c in self.children]

        return self.children[np.argmax(choices_weights)]

    def tree_policy(self):
        '''Select a node to run rollout
        '''
        current_node = self
        dead_end = False
        c=0
        while dead_end == False:
            if not len(current_node._untried_edges)==0: #Try possible moves
                return current_node.expand(), dead_end #Return the expansion
            else: #Select the best move - expand from there
                if len(current_node.children)>0:
                    current_node = current_node.best_child()
                else:
                    dead_end = True

        return current_node, dead_end

    def best_path(self):
        '''Get the best path to take
        '''
        nchains_in_path = 0
        n_total_chains = self.total_chains
        while nchains_in_path<n_total_chains: #add stop if there are no more options
            #Stop search if there are no more untried edges
            v, dead_end = self.tree_policy() #Returns expansion as long as there are actions
            if v:
                if dead_end==True:
                    nchains_in_path = n_total_chains
                else:
                    #Otherwise the best child is returned and the path is thus continued from there.
                    rollout_score = v.rollout() #Rollout
                    v.back_prop(rollout_score) #Backpropagate the score
                    nchains_in_path = len(v.path) #Check path len
                    print(v.path)

        return v


if __name__ == "__main__":
    # pdb_file = r"N:\08_NK_structure_prediction\XL_complex_structure\data\6PBG.pdb"
    # pdb_chains, chain_coords, chain_CA_inds, chain_CB_inds = read_pdb(pdb_file)
    pdb_file = r"N:\08_NK_structure_prediction\XL_complex_structure\data\copa_cope_arfgap3_2_model.cif"
    pdb_chains, chain_coords, chain_CA_inds, chain_CB_inds = read_cif(pdb_file)
    print(chain_coords["A"][:10])
    # print(chain_coords)
    # print(chain_CA_inds)
    # print(chain_CB_inds)