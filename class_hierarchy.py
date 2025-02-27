import numpy as np
import types
from sklearn.metrics import average_precision_score



class ClassHierarchy(object):

    
    def __init__(self, parents, children):

        
        object.__init__(self)
        self.parents = parents
        self.children = children
        self.nodes = set(self.parents.keys()) | set(self.children.keys())
        
        self._depths = { False : {}, True : {} }
        self._hyp_depth_cache = { False : {}, True : {} }
        self._hyp_dist_cache = {}
        self._lcs_cache = {}
        self._wup_cache = {}
        
        self._compute_heights()
        self.max_height = max(self.heights.values())
    
    
    def _compute_heights(self):

        
        def height(id):
            
            if id not in self.heights:
                self.heights[id] = 1 + max((height(child) for child in self.children[id]), default = -1) if id in self.children else 0
            return self.heights[id]
        
        self.heights = {}
        for node in self.nodes:
            height(node)
    
    
    def is_tree(self):

        
        return all(len(parents) <= 1 for parents in self.parents.values())
    
    
    def all_hypernym_depths(self, id, use_min_depth = False):

        
        if id not in self._hyp_depth_cache[use_min_depth]:
            
            depths = {}
            if (id not in self.parents) or (len(self.parents[id]) == 0):
                depths[id] = 1 # root nodes have depth 1
            else:
                for parent in self.parents[id]:
                    for hyp, depth in self.all_hypernym_depths(parent, use_min_depth).items():
                        depths[hyp] = depth
                depths[id] = 1 + min(depths[p] for p in self.parents[id]) if use_min_depth else 1 + max(depths[p] for p in self.parents[id])
            
            self._hyp_depth_cache[use_min_depth][id] = depths
            self._depths[use_min_depth][id] = depths[id]
        
        return self._hyp_depth_cache[use_min_depth][id]
    
    
    def all_hypernym_distances(self, id):

        
        if id not in self._hyp_dist_cache:
        
            distances = { id : 0 }
            if id in self.parents:
                for parent in self.parents[id]:
                    for hyp, dist in self.all_hypernym_distances(parent).items():
                        if (hyp not in distances) or (dist + 1 < distances[hyp]):
                            distances[hyp] = dist + 1

            self._hyp_dist_cache[id] = distances
        
        return self._hyp_dist_cache[id]
    
    
    def root_paths(self, id):

        
        paths = []
        if id in self.parents:
            for parent in self.parents[id]:
                parent_paths = self.root_paths(parent)
                if len(parent_paths) == 0:
                    paths.append([parent])
                else:
                    for parent_path in parent_paths:
                        paths.append([parent] + parent_path)
        return paths
    
    
    def lcs(self, a, b, use_min_depth = False):

        
        if (a,b) not in self._lcs_cache:
        
            hypernym_depths = self.all_hypernym_depths(a, use_min_depth)
            common_hypernyms = set(hypernym_depths.keys()) & set(self.all_hypernym_depths(b, use_min_depth).keys())

            self._lcs_cache[(a,b)] = self._lcs_cache[(b,a)] = max(common_hypernyms, key = lambda hyp: hypernym_depths[hyp], default = None)
        
        return self._lcs_cache[(a,b)]
    
    
    def shortest_path_length(self, a, b):

        
        dist1 = self.all_hypernym_distances(a)
        dist2 = self.all_hypernym_distances(b)
        common_hypernyms = set(dist1.keys()) & set(dist2.keys())
        
        return min((dist1[hyp] + dist2[hyp] for hyp in common_hypernyms), default = None)
    
    
    def depth(self, id, use_min_depth = False):

        
        if id not in self._depths[use_min_depth]:
            
            if (not id in self.parents) or (len(self.parents[id]) == 0):
                self._depths[use_min_depth][id] = 1 # root nodes have depth 1
            else:
                parent_depths = (self.depth(p, use_min_depth) for p in self.parents[id])
                self._depths[use_min_depth][id] = 1 + min(parent_depths) if use_min_depth else 1 + max(parent_depths)
        
        return self._depths[use_min_depth][id]
    
    
    def wup_similarity(self, a, b):

        
        if (a,b) not in self._wup_cache:
        
            lcs = self.lcs(a, b)
            ds = self.depth(lcs)
            d1 = ds + self.shortest_path_length(a, lcs)
            d2 = ds + self.shortest_path_length(b, lcs)
            self._wup_cache[(a,b)] = self._wup_cache[(b,a)] = (2.0 * ds) / (d1 + d2)
        
        return self._wup_cache[(a,b)]
    
    
    def lcs_height(self, a, b):
 
        return self.heights[self.lcs(a, b)] / self.max_height
    
    
    def hierarchical_precision(self, retrieved, labels, ks = [1, 10, 50, 100], compute_ahp = False, compute_ap = False, ignore_qids = True, all_ids = None):
        
        
        if isinstance(ks, int):
            ks = [ks]
        kmax = max(ks)
        if not isinstance(compute_ahp, bool):
            kmax = max(kmax, int(compute_ahp))
        
        prec = { 'P@{} ({})'.format(k, type) : {} for k in ks for type in ('WUP', 'LCS_HEIGHT') }
        if compute_ahp:
            ahp_suffix = '' if isinstance(compute_ahp, bool) else '@{}'.format(compute_ahp)
            prec['AHP{} (WUP)'.format(ahp_suffix)] = {}
            prec['AHP{} (LCS_HEIGHT)'.format(ahp_suffix)] = {}
        if compute_ap:
            prec['AP'] = {}
        
        best_wup_cum = {}
        best_lcs_cum = {}
        
        for qid, ret in (retrieved if isinstance(retrieved, types.GeneratorType) else retrieved.items()):
            
            lbl = labels[qid]
            
            # Append missing images to the end of the ranking for proper determination of the optimal ranking
            if all_ids and (len(ret) < len(all_ids)):
                sret = set(ret)
                ret = ret + [id for id in all_ids if id not in sret]
            
            # Compute WUP similarity and determine optimal ranking for this label
            if (lbl not in best_wup_cum) or (compute_ahp is True):
                # We inlined the cache lookup from self.wup_similarity() here to reduce unnecessary function calls.
                wup = [self._wup_cache[(lbl, labels[r])] if (lbl, labels[r]) in self._wup_cache else self.wup_similarity(lbl, labels[r]) for r in ret]
                if lbl not in best_wup_cum:
                    best_wup_cum[lbl] = np.cumsum(sorted(wup, reverse = True))
            else:
                wup = [self._wup_cache[(lbl, labels[r])] if (lbl, labels[r]) in self._wup_cache else self.wup_similarity(lbl, labels[r]) for r in ret[:kmax+1]]
            
            # Compute LCS height based similarity and determine optimal ranking for this label
            if (lbl not in best_lcs_cum) or (compute_ahp is True):
                # We inline self.lcs_height() here to reduce function calls.
                # We also don't need to check whether the class pair is cached in self._lcs_cache, since we computed the WUP before which does that implicitly.
                lcs = (1.0 - np.array([self.heights[self._lcs_cache[(lbl, labels[r])]] for r in ret]) / self.max_height).tolist()
                if lbl not in best_lcs_cum:
                    best_lcs_cum[lbl] = np.cumsum(sorted(lcs, reverse = True))
            else:
                lcs = (1.0 - np.array([self.heights[self._lcs_cache[(lbl, labels[r])]] for r in ret[:kmax+1]]) / self.max_height).tolist()
            
            # Remove query from retrieval list
            cum_best_wup = best_wup_cum[lbl]
            cum_best_lcs = best_lcs_cum[lbl]
            if ignore_qids:
                try:
                    qid_ind = ret.index(qid)
                    if qid_ind < len(wup):
                        del wup[qid_ind]
                        del lcs[qid_ind]
                        cum_best_wup = np.concatenate((cum_best_wup[:qid_ind], cum_best_wup[qid_ind+1:] - 1.0))
                        cum_best_lcs = np.concatenate((cum_best_lcs[:qid_ind], cum_best_lcs[qid_ind+1:] - 1.0))
                except ValueError:
                    pass
            
            # Compute hierarchical precision for several cut-off points
            for k in ks:
                prec['P@{} (WUP)'.format(k)][qid]        = sum(wup[:k]) / cum_best_wup[k-1]
                prec['P@{} (LCS_HEIGHT)'.format(k)][qid] = sum(lcs[:k]) / cum_best_lcs[k-1]
            if compute_ahp:
                if isinstance(compute_ahp, bool):
                    prec['AHP (WUP)'][qid]        = np.mean(np.cumsum(wup) / cum_best_wup) - (wup[0] / cum_best_wup[0] + wup[-1] / cum_best_wup[-1]) / (2 * len(wup))
                    prec['AHP (LCS_HEIGHT)'][qid] = np.mean(np.cumsum(lcs) / cum_best_lcs) - (lcs[0] / cum_best_lcs[0] + lcs[-1] / cum_best_lcs[-1]) / (2 * len(lcs))
                else:
                    prec['AHP{} (WUP)'.format(ahp_suffix)][qid]        = np.mean(np.cumsum(wup[:compute_ahp]) / cum_best_wup[:compute_ahp]) - (wup[0] / cum_best_wup[0] + wup[compute_ahp-1] / cum_best_wup[compute_ahp-1]) / (2 * compute_ahp)
                    prec['AHP{} (LCS_HEIGHT)'.format(ahp_suffix)][qid] = np.mean(np.cumsum(lcs[:compute_ahp]) / cum_best_lcs[:compute_ahp]) - (lcs[0] / cum_best_lcs[0] + lcs[compute_ahp-1] / cum_best_lcs[compute_ahp-1]) / (2 * compute_ahp)
            if compute_ap:
                prec['AP'][qid] = average_precision_score(
                    [labels[r] == lbl for r in ret if (not ignore_qids) or (r != qid)],
                    [-i for i, r in enumerate(ret) if (not ignore_qids) or (r != qid)]
                )
        
        return { metric : sum(values.values()) / len(values) for metric, values in prec.items() }, prec
    
    
    def save(self, filename, is_a_relations = False):
        """ Writes the hierarchy structure to a text file as lines of parent-child or child-parent tuples.
        
        filename - Path to the file to be written.
        is_a_relations - If set to `True`, the hierarchy will be exported is child-parent tuples, otherwise as parent-child tuples.
        """
        
        with open(filename, 'w') as f:
            if is_a_relations:
                for child, parents in self.parents.items():
                    for parent in parents:
                        f.write('{} {}\n'.format(child, parent))
            else:
                for parent, children in self.children.items():
                    for child in children:
                        f.write('{} {}\n'.format(parent, child))
    
    
    @classmethod
    def from_file(cls, rel_file, is_a_relations = False, id_type = str):
        """ Constructs a class hierarchy based on a file with parent-child relations.
        
        rel_file - Path to a file specifying the relations between elements in the hierarchy, given by lines of ID tuples.
        is_a_relations - If set to `True`, `rel_file` is supposed to contain `<child> <parent>` tuples, otherwise `<parent> <child>` tuples.
        id_type - Data type of element IDs.
        
        Returns: a new ClassHierarchy instance
        """
        
        parents, children = {}, {}
        with open(rel_file) as f:
            for l in f:
                if l.strip() != '':
                    
                    parent, child = [id_type(id) for id in l.strip().split(maxsplit = 1)]
                    if is_a_relations:
                        parent, child = child, parent
                    
                    if child in parents:
                        parents[child].append(parent)
                    else:
                        parents[child] = [parent]
                    
                    if parent in children:
                        children[parent].append(child)
                    else:
                        children[parent] = [child]
        
        return cls(parents, children)
