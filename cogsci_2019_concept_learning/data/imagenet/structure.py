from collections import Counter
from nltk.corpus import wordnet as wn
import numpy as np

from cogsci_2019_concept_learning.utils.utils_gen import log_function_call


class ImageNetStructure(object):
    """TODO"""

    def __getattribute__(self, name):
        """Lazily initialize the required and related attributes."""

        check_names = ['train_labels_to_leaf_labels_map', 'val_labels_to_leaf_labels_map', 'test_labels_to_leaf_labels_map', 'label_to_negative_examples_map']

        if name in check_names and super().__getattribute__(name) is None:

            train_leaf_labels = list(self.train_label_to_image_map.keys())
            val_leaf_labels   = list(self.val_label_to_image_map.keys())
            test_leaf_labels  = list(self.test_label_to_image_map.keys())

            assert not set(train_leaf_labels).intersection(val_leaf_labels)
            assert not set(train_leaf_labels).intersection(test_leaf_labels)
            assert not set(val_leaf_labels).intersection(test_leaf_labels)

            train_map, val_map, test_map, negative_map = self.create_split(train_leaf_labels, val_leaf_labels, test_leaf_labels)

            self.train_labels_to_leaf_labels_map = train_map
            self.val_labels_to_leaf_labels_map   = val_map
            self.test_labels_to_leaf_labels_map  = test_map
            self.label_to_negative_examples_map  = negative_map

        return super().__getattribute__(name)


class FlatImageNet(ImageNetStructure):
    """TODO"""

    @staticmethod
    def create_split(train_leaf_nodes, val_leaf_nodes, test_leaf_nodes):
        """TODO"""

        # Flat dataset -- just map leaves to leaves
        train_nodes_to_leaves_map = {leaf: [leaf] for leaf in train_leaf_nodes}
        val_nodes_to_leaves_map   = {leaf: [leaf] for leaf in val_leaf_nodes}
        test_nodes_to_leaves_map  = {leaf: [leaf] for leaf in test_leaf_nodes}

        node_to_negative_leaves_map = {}
        node_to_negative_leaves_map.update({leaf: list(set(train_leaf_nodes) - set([leaf])) for leaf in train_leaf_nodes})
        node_to_negative_leaves_map.update({leaf: list(set(val_leaf_nodes)   - set([leaf])) for leaf in val_leaf_nodes})
        node_to_negative_leaves_map.update({leaf: list(set(test_leaf_nodes)  - set([leaf])) for leaf in test_leaf_nodes})

        return train_nodes_to_leaves_map, val_nodes_to_leaves_map, test_nodes_to_leaves_map, node_to_negative_leaves_map


class HierarchicalImageNet(ImageNetStructure):
    """TODO"""

    def get_node_to_leaf_map(self, hypernyms, leaf_nodes):
        """TODO"""

        # Add internal nodes
        hypernym_to_leaf_map = {}
        for node in hypernyms:
            hypernym_leaf_list = []
            for leaf in self.get_hyponym_lists([node]):
                if leaf in leaf_nodes:
                    hypernym_leaf_list += [leaf]

            # Include node if the dominated leaf list is smaller than some constant
            # This serves to exclude nodes that are too abstract
            if 1 < len(hypernym_leaf_list) < 9:  # TODO: upper bound is a hyperparameter
                hypernym_to_leaf_map[node] = hypernym_leaf_list

        # Consolidate the indistinguishable concepts (i.e., keep only
        # hypernyms that map to a unique list of leaf nodes)
        seen = []
        new_hypernym_to_leaf_map = {}
        for (k, v) in hypernym_to_leaf_map.items():
            if str(v) not in seen:
                seen += [str(v)]
                new_hypernym_to_leaf_map[k] = v

        hypernym_to_leaf_map = new_hypernym_to_leaf_map

        # Add leaf nodes
        for leaf in leaf_nodes:
            hypernym_to_leaf_map[leaf] = [leaf]

        return hypernym_to_leaf_map

    def get_closures(self, wnids, closure_fn, max_depth=None):
        """TODO"""
        synsets  = [wn._synset_from_pos_and_offset('n', int(x[1:])) for x in wnids]
        if max_depth is not None:
            closures = list(set(['n%08d' % x.offset() for y in synsets for x in y.closure(closure_fn, depth=max_depth)]))
        else:
            closures = list(set(['n%08d' % x.offset() for y in synsets for x in y.closure(closure_fn)]))
        return closures

    def get_hypernym_lists(self, wnids):
        """TODO"""
        return self.get_closures(wnids, lambda s: s.hypernyms())

    def get_hyponym_lists(self, wnids):
        """TODO"""
        return self.get_closures(wnids, lambda s: s.hyponyms())

    @log_function_call("hierarchical dataset construction")
    def create_split(self, train_leaf_nodes, val_leaf_nodes, test_leaf_nodes):
        """TODO"""

        # Get all hypernyms corresponding to the leaf nodes for each set
        train_hypernym_closure = self.get_hypernym_lists(train_leaf_nodes)
        val_hypernym_closure   = self.get_hypernym_lists(val_leaf_nodes)
        test_hypernym_closure  = self.get_hypernym_lists(test_leaf_nodes)

        # Investigate pairwise intersections; get rid of hypernym nodes that overlap
        intersection = set(train_hypernym_closure).intersection(val_hypernym_closure) |\
                       set(test_hypernym_closure).intersection(val_hypernym_closure) |\
                       set(train_hypernym_closure).intersection(test_hypernym_closure)

        train_hypernym_closure = [x for x in train_hypernym_closure if x not in intersection]
        val_hypernym_closure   = [x for x in val_hypernym_closure   if x not in intersection]
        test_hypernym_closure  = [x for x in test_hypernym_closure  if x not in intersection]

        # Get hyponyms under the remaining nodes...
        # Note that these closures may be strictly smaller than the
        # corresponding set of leaf nodes, since there may not be a dominating
        # hypernym for each leaf node that exists in only one of the
        # train / val / test sets.
        train_hyponym_closure  = [y for y in self.get_hyponym_lists(train_hypernym_closure) if y in train_leaf_nodes]
        val_hyponym_closure    = [y for y in self.get_hyponym_lists(val_hypernym_closure)   if y in val_leaf_nodes]
        test_hyponym_closure   = [y for y in self.get_hyponym_lists(test_hypernym_closure)  if y in test_leaf_nodes]

        # ...and remove nodes if they dominate the same children.
        # We must do this because ImageNet is a DAG, not a tree.
        pruned_train_hypernyms = [x for x in train_hypernym_closure if not set([y for y in self.get_hyponym_lists([x]) if y in train_leaf_nodes]).intersection(val_hyponym_closure + test_hyponym_closure)]
        pruned_val_hypernyms   = [x for x in val_hypernym_closure   if not set([y for y in self.get_hyponym_lists([x]) if y in val_leaf_nodes]).intersection(train_hyponym_closure + test_hyponym_closure)]
        pruned_test_hypernyms  = [x for x in test_hypernym_closure  if not set([y for y in self.get_hyponym_lists([x]) if y in test_leaf_nodes]).intersection(train_hyponym_closure + val_hyponym_closure)]

        train_nodes_to_leaves_map = self.get_node_to_leaf_map(pruned_train_hypernyms, train_leaf_nodes)
        val_nodes_to_leaves_map   = self.get_node_to_leaf_map(pruned_val_hypernyms,   val_leaf_nodes)
        test_nodes_to_leaves_map  = self.get_node_to_leaf_map(pruned_test_hypernyms,  test_leaf_nodes)

        # Sanity check that there are no overlaps
        assert not set(train_nodes_to_leaves_map).intersection(val_nodes_to_leaves_map)
        assert not set(train_nodes_to_leaves_map).intersection(test_nodes_to_leaves_map)
        assert not set(val_nodes_to_leaves_map).intersection(test_nodes_to_leaves_map)
        assert not set([x for y in train_nodes_to_leaves_map.values() for x in y]).intersection([x for y in val_nodes_to_leaves_map.values()  for x in y])
        assert not set([x for y in train_nodes_to_leaves_map.values() for x in y]).intersection([x for y in test_nodes_to_leaves_map.values() for x in y])
        assert not set([x for y in val_nodes_to_leaves_map.values()   for x in y]).intersection([x for y in test_nodes_to_leaves_map.values() for x in y])

        # Get negative examples.
        # This dictionary can be shared between train and val if we are careful which negatives we associate with each node, since the nodes themselves are disjoint.
        # As a simplification, just sample uniformly from the complement set of leaf nodes (within the same train/val/test set) (i.e., do not sample negative concepts).
        node_to_negative_leaves_map = {}
        node_to_negative_leaves_map.update(self.get_node_to_negatives_map(train_leaf_nodes, train_nodes_to_leaves_map))
        node_to_negative_leaves_map.update(self.get_node_to_negatives_map(val_leaf_nodes,   val_nodes_to_leaves_map))
        node_to_negative_leaves_map.update(self.get_node_to_negatives_map(test_leaf_nodes,  test_nodes_to_leaves_map))

        return train_nodes_to_leaves_map, val_nodes_to_leaves_map, test_nodes_to_leaves_map, node_to_negative_leaves_map

    def get_node_to_negatives_map(self, leaf_nodes, nodes_to_leaves_map):
        return {node: list(set([leaf for node_in_complement in list(set(leaf_nodes) - set(nodes_to_leaves_map[node])) for leaf in nodes_to_leaves_map[node_in_complement]])) for node in nodes_to_leaves_map.keys()}


class InternalNodeImageNet(HierarchicalImageNet):
    """TODO"""

    def get_node_to_leaf_map(self, hypernyms, leaf_nodes):
        """TODO"""

        # Add internal nodes
        hypernym_to_leaf_map = {}
        for node in hypernyms:
            hypernym_leaf_list = []
            for leaf in self.get_hyponym_lists([node]):
                if leaf in leaf_nodes:
                    hypernym_leaf_list += [leaf]

            # Include node if the dominated leaf list is smaller than some constant
            # This serves to exclude nodes that are too abstract
            if 1 < len(hypernym_leaf_list) < 9:  # TODO: upper bound is a hyperparameter
                hypernym_to_leaf_map[node] = hypernym_leaf_list

        # Consolidate the indistinguishable concepts (i.e., keep only
        # hypernyms that map to a unique list of leaf nodes)
        seen = []
        new_hypernym_to_leaf_map = {}
        for (k, v) in hypernym_to_leaf_map.items():
            if str(v) not in seen:
                seen += [str(v)]
                new_hypernym_to_leaf_map[k] = v

        hypernym_to_leaf_map = new_hypernym_to_leaf_map

        # Return without adding leaf nodes
        return hypernym_to_leaf_map

    def get_node_to_negatives_map(self, leaf_nodes, nodes_to_leaves_map):

        # Augment node_to_leaves_map with leaves
        copy_nodes_to_leaves_map = nodes_to_leaves_map.copy()
        for leaf in leaf_nodes:
            copy_nodes_to_leaves_map[leaf] = [leaf]

        return {node: list(set([leaf for node_in_complement in list(set(leaf_nodes) - set(copy_nodes_to_leaves_map[node])) for leaf in copy_nodes_to_leaves_map[node_in_complement]])) for node in copy_nodes_to_leaves_map.keys()}


class RandomImageNet(ImageNetStructure):
    """TODO"""

    @classmethod
    def create_split(cls, train_leaf_nodes, val_leaf_nodes, test_leaf_nodes):
        """TODO"""
        train_nodes_to_leaves_map = {}
        val_nodes_to_leaves_map   = {}
        test_nodes_to_leaves_map  = {}

        # An arbitrary index for random concepts
        idx = 0

        # Randomly glue nodes together for meta-training set
        for num_leaves, num_classes in cls.num_leaves_to_classes_train.items():
            if num_leaves == 1:
                sampled_nodes = [[x] for x in train_leaf_nodes]
            else:
                sampled_nodes = [list(np.random.choice(train_leaf_nodes, size=num_leaves, replace=False)) for _ in range(num_classes)]
            for leaf_list in sampled_nodes:
                train_nodes_to_leaves_map[str(idx)] = leaf_list
                idx += 1

        # Randomly glue nodes together for meta-validation set
        for num_leaves, num_classes in cls.num_leaves_to_classes_val.items():
            if num_leaves == 1:
                sampled_nodes = [[x] for x in val_leaf_nodes]
            else:
                sampled_nodes = [list(np.random.choice(val_leaf_nodes, size=num_leaves, replace=False)) for _ in range(num_classes)]
            for leaf_list in sampled_nodes:
                val_nodes_to_leaves_map[str(idx)] = leaf_list
                idx += 1

        # Randomly glue nodes together for meta-test set
        for num_leaves, num_classes in cls.num_leaves_to_classes_test.items():
            if num_leaves == 1:
                sampled_nodes = [[x] for x in test_leaf_nodes]
            else:
                sampled_nodes = [list(np.random.choice(test_leaf_nodes, size=num_leaves, replace=False)) for _ in range(num_classes)]
            for leaf_list in sampled_nodes:
                test_nodes_to_leaves_map[str(idx)] = leaf_list
                idx += 1

        node_to_negative_leaves_map = {}
        node_to_negative_leaves_map.update({k: list(set(train_leaf_nodes) - set(v)) for k, v in train_nodes_to_leaves_map.items()})
        node_to_negative_leaves_map.update({k: list(set(val_leaf_nodes)   - set(v)) for k, v in val_nodes_to_leaves_map.items()})
        node_to_negative_leaves_map.update({k: list(set(test_leaf_nodes)  - set(v)) for k, v in test_nodes_to_leaves_map.items()})

        return train_nodes_to_leaves_map, val_nodes_to_leaves_map, test_nodes_to_leaves_map, node_to_negative_leaves_map


class BalancedDataset(object):
    """TODO"""

    @staticmethod
    def normalize_counts(node_to_leaves_map):
        counts = Counter([len(leaves) for leaves in node_to_leaves_map.values()])
        ratio = max(2, counts[1] // counts[2])
        new_node_to_leaves_map = {}
        for node, leaves in node_to_leaves_map.items():
            if len(leaves) > 1:
                new_node_to_leaves_map.update({'%s_%d' % (node, i): leaves for i in range(ratio)})
            else:
                new_node_to_leaves_map[node] = leaves

        return new_node_to_leaves_map

    def create_split(self, train_leaf_nodes, val_leaf_nodes, test_leaf_nodes):
        train_nodes_to_leaves_map, val_nodes_to_leaves_map, test_nodes_to_leaves_map, node_to_negative_leaves_map = super().create_split(train_leaf_nodes, val_leaf_nodes, test_leaf_nodes)

        new_train_nodes_to_leaves_map = self.normalize_counts(train_nodes_to_leaves_map)
        new_val_nodes_to_leaves_map = self.normalize_counts(val_nodes_to_leaves_map)
        new_test_nodes_to_leaves_map = self.normalize_counts(test_nodes_to_leaves_map)

        new_node_to_negative_leaves_map = {}
        new_node_to_negative_leaves_map.update(self.get_node_to_negatives_map(train_leaf_nodes, new_train_nodes_to_leaves_map))
        new_node_to_negative_leaves_map.update(self.get_node_to_negatives_map(val_leaf_nodes,   new_val_nodes_to_leaves_map))
        new_node_to_negative_leaves_map.update(self.get_node_to_negatives_map(test_leaf_nodes,  new_test_nodes_to_leaves_map))

        return new_train_nodes_to_leaves_map, new_val_nodes_to_leaves_map, new_test_nodes_to_leaves_map, new_node_to_negative_leaves_map


class BalancedHierarchicalImageNet(HierarchicalImageNet, BalancedDataset):
    """TODO Fixes the leaf class imbalance."""
    pass


class BalancedRandomImageNet(RandomImageNet, BalancedDataset):
    """TODO Fixes the leaf class imbalance."""
