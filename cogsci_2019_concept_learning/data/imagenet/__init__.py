import os
import pickle

from cogsci_2019_concept_learning.data.imagenet.structure import FlatImageNet, BalancedHierarchicalImageNet, BalancedRandomImageNet, HierarchicalImageNet, RandomImageNet, InternalNodeImageNet
from cogsci_2019_concept_learning.data.imagenet.splits import ILSVRCSplit, OriolMiniImageNetSplit, SachinMiniImageNetSplit
from cogsci_2019_concept_learning.data.imagenet.task import *


IMAGENET_DIR = os.path.join(os.path.dirname(__file__))

with open(os.path.join(IMAGENET_DIR, 'ilsvrc2012_synsets.pkl'), 'rb') as f:
    ilsvrc2012_synsets = pickle.load(f)

try:
    with open(os.path.join(IMAGENET_DIR, 'synset_to_hyponyms_map.pkl'), 'rb') as f:
        imagenet_synset_to_hyponyms_map = pickle.load(f)

except FileNotFoundError:
    # Harvest from the web
    raise NotImplementedError
    synset_list_url = "http://www.image-net.org/api/text/imagenet.synset.obtain_synset_list"
    synset_list = list(filter(None, urllib.request.urlopen(synset_list_url).read().decode('utf-8').split('\n')))

    hyponym_list_url = "http://www.image-net.org/api/text/wordnet.structure.hyponym?wnid={%s}&full=1"
    hyponym_access = lambda x: list(filter(urllib.request.urlopen(hyponym_list_url % x).read().decode("utf-8").rstrip('\r\n').split('\r\n-')))

    for synset in synset_list:
        hyponyms = hyponym_access(synset)
        synset_to_hyponyms_map[hyponyms[0]] = hyponyms[1:]

    # Dump to file
    imagenet_synset_to_hyponyms_map = {}
    with open(os.path.join(source_dir, 'synset_to_hyponyms_map.pkl'), 'wb') as f:
        pickle.dump(imagenet_synset_to_hyponyms_map, f, pickle.HIGHEST_PROTOCOL)

with open(os.path.join(IMAGENET_DIR, 'rosch_to_imagenet_map.pkl'), 'rb') as f:
    rosch_to_imagenet_map = pickle.load(f)


class FlatILSVRC(FlatImageNet, ILSVRCSplit):
    """TODO"""

    # Shared among subclasses of this class
    train_labels_to_leaf_labels_map = None
    val_labels_to_leaf_labels_map   = None
    test_labels_to_leaf_labels_map  = None
    label_to_negative_examples_map  = None


class FlatOriolMiniImageNet(FlatImageNet, OriolMiniImageNetSplit):
    """TODO"""

    # Shared among subclasses of this class
    train_labels_to_leaf_labels_map = None
    val_labels_to_leaf_labels_map   = None
    test_labels_to_leaf_labels_map  = None
    label_to_negative_examples_map  = None


class FlatSachinMiniImageNet(FlatImageNet, SachinMiniImageNetSplit):
    """TODO"""

    # Shared among subclasses of this class
    train_labels_to_leaf_labels_map = None
    val_labels_to_leaf_labels_map   = None
    test_labels_to_leaf_labels_map  = None
    label_to_negative_examples_map  = None


class InternalNodeILSVRC(InternalNodeImageNet, ILSVRCSplit):
    """TODO"""

    # Shared among subclasses of this class
    train_labels_to_leaf_labels_map = None
    val_labels_to_leaf_labels_map   = None
    test_labels_to_leaf_labels_map  = None
    label_to_negative_examples_map  = None


class HierarchicalILSVRC(HierarchicalImageNet, ILSVRCSplit):
    """TODO"""

    # Shared among subclasses of this class
    train_labels_to_leaf_labels_map = None
    val_labels_to_leaf_labels_map   = None
    test_labels_to_leaf_labels_map  = None
    label_to_negative_examples_map  = None


class BalancedHierarchicalILSVRC(BalancedHierarchicalImageNet, ILSVRCSplit):
    """TODO"""

    # Shared among subclasses of this class
    train_labels_to_leaf_labels_map = None
    val_labels_to_leaf_labels_map   = None
    test_labels_to_leaf_labels_map  = None
    label_to_negative_examples_map  = None


class BalancedRandomILSVRC(BalancedRandomImageNet, ILSVRCSplit):
    """TODO"""

    # Shared among subclasses of this class
    train_labels_to_leaf_labels_map = None
    val_labels_to_leaf_labels_map   = None
    test_labels_to_leaf_labels_map  = None
    label_to_negative_examples_map  = None

    # Specify distribution to align with corresponding hierarchical class
    num_leaves_to_classes_train = {
        1: 494,
        2: 72,
        3: 30,
        4: 20,
        5: 9,
        6: 7,
        7: 10,
        8: 6,
    }
    num_leaves_to_classes_val = {
        1: 193,
        2: 35,
        3: 10,
        4: 1,
        5: 2,
        6: 4,
        8: 2,
    }
    num_leaves_to_classes_test = {
        1: 223,
        2: 36,
        3: 20,
        4: 14,
        5: 4,
        6: 2,
        7: 4,
        8: 2,
    }


class HierarchicalOriolMiniImageNet(HierarchicalImageNet, OriolMiniImageNetSplit):
    """TODO"""

    # Shared among subclasses of this class
    train_labels_to_leaf_labels_map = None
    val_labels_to_leaf_labels_map   = None
    test_labels_to_leaf_labels_map  = None
    label_to_negative_examples_map  = None


class HierarchicalSachinMiniImageNet(HierarchicalImageNet, SachinMiniImageNetSplit):
    """TODO"""

    # Shared among subclasses of this class
    train_labels_to_leaf_labels_map = None
    val_labels_to_leaf_labels_map   = None
    test_labels_to_leaf_labels_map  = None
    label_to_negative_examples_map  = None

    def __init__(self, *args, **kwargs):
        raise NotImplementedError("Cannot make validation nor test sets disjoint from training and maintain hierarchical structure.")


class RandomILSVRC(RandomImageNet, ILSVRCSplit):
    """TODO"""

    # Shared among subclasses of this class
    train_labels_to_leaf_labels_map = None
    val_labels_to_leaf_labels_map   = None
    test_labels_to_leaf_labels_map  = None
    label_to_negative_examples_map  = None

    # Specify distribution to align with correspdoning hierarchical class
    num_leaves_to_classes_train = {
        1: 494,
        2: 72,
        3: 30,
        4: 20,
        5: 9,
        6: 7,
        7: 10,
        8: 6,
    }
    num_leaves_to_classes_val = {
        1: 193,
        2: 35,
        3: 10,
        4: 1,
        5: 2,
        6: 4,
        8: 2,
    }
    num_leaves_to_classes_test = {
        1: 223,
        2: 36,
        3: 20,
        4: 14,
        5: 4,
        6: 2,
        7: 4,
        8: 2,
    }


class RandomOriolMiniImageNet(RandomImageNet, OriolMiniImageNetSplit):
    """TODO"""

    # Shared among subclasses of this class
    train_labels_to_leaf_labels_map = None
    val_labels_to_leaf_labels_map   = None
    test_labels_to_leaf_labels_map  = None
    label_to_negative_examples_map  = None

    # Specify distribution to align with corresponding hierarchical class
    num_leaves_to_classes_train = {
        1: 80,
        2: 20,
        3: 10,
    }
    num_leaves_to_classes_val = {
        1: 20,
        2: 5,
        3: 5,
    }
    num_leaves_to_classes_test = {
        1: 20,
        2: 5,
        3: 5,
    }


class RandomSachinMiniImageNet(RandomImageNet, SachinMiniImageNetSplit):
    """TODO"""
    pass


class FlatUnaryILSVRC(FlatILSVRC, UnaryClassification):
    name = 'flat_unary_ilsvrc'


class FlatFiveToOneBinaryILSVRC(FlatILSVRC, FiveToOneBinaryClassification):
    name = 'flat_5-1binary_ilsvrc'


class FlatFiveToTwoBinaryILSVRC(FlatILSVRC, FiveToTwoBinaryClassification):
    name = 'flat_5-2binary_ilsvrc'


class FlatFiveToThreeBinaryILSVRC(FlatILSVRC, FiveToThreeBinaryClassification):
    name = 'flat_5-3binary_ilsvrc'


class FlatFiveToFourBinaryILSVRC(FlatILSVRC, FiveToFourBinaryClassification):
    name = 'flat_5-4binary_ilsvrc'


class FlatOneToOneBinaryILSVRC(FlatILSVRC, OneToOneBinaryClassification):
    name = 'flat_1-1binary_ilsvrc'


class FlatMultiWayILSVRC(FlatILSVRC, MultiWayClassification):
    name = 'flat_multiway_ilsvrc'


class HierarchicalUnaryILSVRC(HierarchicalILSVRC, UnaryClassification):
    name = 'hier_unary_ilsvrc'


class HierarchicalFiveToOneBinaryILSVRC(HierarchicalILSVRC, FiveToOneBinaryClassification):
    name = 'hier_5-1binary_ilsvrc'


class HierarchicalFiveToTwoBinaryILSVRC(HierarchicalILSVRC, FiveToTwoBinaryClassification):
    name = 'hier_5-2binary_ilsvrc'


class HierarchicalFiveToThreeBinaryILSVRC(HierarchicalILSVRC, FiveToThreeBinaryClassification):
    name = 'hier_5-3binary_ilsvrc'


class HierarchicalFiveToFourBinaryILSVRC(HierarchicalILSVRC, FiveToFourBinaryClassification):
    name = 'hier_5-4binary_ilsvrc'


class HierarchicalOneToOneBinaryILSVRC(HierarchicalILSVRC, OneToOneBinaryClassification):
    name = 'hier_1-1binary_ilsvrc'


class HierarchicalMultiWayILSVRC(HierarchicalILSVRC, MultiWayClassification):
    name = 'hier_multiway_ilsvrc'


class BalancedHierarchicalUnaryILSVRC(BalancedHierarchicalILSVRC, UnaryClassification):
    name = 'balanced_hier_unary_ilsvrc'


class BalancedHierarchicalFiveToOneBinaryILSVRC(BalancedHierarchicalILSVRC, FiveToOneBinaryClassification):
    name = 'balanced_hier_5-1binary_ilsvrc'


class BalancedHierarchicalFiveToTwoBinaryILSVRC(BalancedHierarchicalILSVRC, FiveToTwoBinaryClassification):
    name = 'balanced_hier_5-2binary_ilsvrc'


class BalancedHierarchicalFiveToThreeBinaryILSVRC(BalancedHierarchicalILSVRC, FiveToThreeBinaryClassification):
    name = 'balanced_hier_5-3binary_ilsvrc'


class BalancedHierarchicalFiveToFourBinaryILSVRC(BalancedHierarchicalILSVRC, FiveToFourBinaryClassification):
    name = 'balanced_hier_5-4binary_ilsvrc'


class BalancedHierarchicalOneToOneBinaryILSVRC(BalancedHierarchicalILSVRC, OneToOneBinaryClassification):
    name = 'balanced_hier_1-1binary_ilsvrc'


class BalancedHierarchicalMultiWayILSVRC(BalancedHierarchicalILSVRC, MultiWayClassification):
    name = 'balanced_hier_multiway_ilsvrc'


class InternalNodeUnaryILSVRC(InternalNodeILSVRC, UnaryClassification):
    name = 'internal_unary_ilsvrc'


class InternalNodeFiveToOneBinaryILSVRC(InternalNodeILSVRC, FiveToOneBinaryClassification):
    name = 'internal_5-1binary_ilsvrc'


class InternalNodeFiveToTwoBinaryILSVRC(InternalNodeILSVRC, FiveToTwoBinaryClassification):
    name = 'internal_5-2binary_ilsvrc'


class InternalNodeFiveToThreeBinaryILSVRC(InternalNodeILSVRC, FiveToThreeBinaryClassification):
    name = 'internal_5-3binary_ilsvrc'


class InternalNodeFiveToFourBinaryILSVRC(InternalNodeILSVRC, FiveToFourBinaryClassification):
    name = 'internal_5-4binary_ilsvrc'


class InternalNodeOneToOneBinaryILSVRC(InternalNodeILSVRC, OneToOneBinaryClassification):
    name = 'internal_1-1binary_ilsvrc'


class InternalNodeMultiWayILSVRC(InternalNodeILSVRC, MultiWayClassification):
    name = 'internal_multiway_ilsvrc'


class RandomUnaryILSVRC(RandomILSVRC, UnaryClassification):
    name = 'rand_unary_ilsvrc'


class RandomFiveToOneBinaryILSVRC(RandomILSVRC, FiveToOneBinaryClassification):
    name = 'rand_5-1binary_ilsvrc'


class RandomFiveToTwoBinaryILSVRC(RandomILSVRC, FiveToTwoBinaryClassification):
    name = 'rand_5-2binary_ilsvrc'


class RandomFiveToThreeBinaryILSVRC(RandomILSVRC, FiveToThreeBinaryClassification):
    name = 'rand_5-3binary_ilsvrc'


class RandomFiveToFourBinaryILSVRC(RandomILSVRC, FiveToFourBinaryClassification):
    name = 'rand_5-4binary_ilsvrc'


class RandomOneToOneBinaryILSVRC(RandomILSVRC, OneToOneBinaryClassification):
    name = 'rand_1-1binary_ilsvrc'


class RandomMultiWayILSVRC(RandomILSVRC, MultiWayClassification):
    name = 'rand_multiway_ilsvrc'


class BalancedRandomUnaryILSVRC(BalancedRandomILSVRC, UnaryClassification):
    name = 'balanced_rand_unary_ilsvrc'


class BalancedRandomFiveToOneBinaryILSVRC(BalancedRandomILSVRC, FiveToOneBinaryClassification):
    name = 'balanced_rand_5-1binary_ilsvrc'


class BalancedRandomFiveToTwoBinaryILSVRC(BalancedRandomILSVRC, FiveToTwoBinaryClassification):
    name = 'balanced_rand_5-2binary_ilsvrc'


class BalancedRandomFiveToThreeBinaryILSVRC(BalancedRandomILSVRC, FiveToThreeBinaryClassification):
    name = 'balanced_rand_5-3binary_ilsvrc'


class BalancedRandomFiveToFourBinaryILSVRC(BalancedRandomILSVRC, FiveToFourBinaryClassification):
    name = 'balanced_rand_5-4binary_ilsvrc'


class BalancedRandomOneToOneBinaryILSVRC(BalancedRandomILSVRC, OneToOneBinaryClassification):
    name = 'balanced_rand_1-1binary_ilsvrc'


class BalancedRandomMultiWayILSVRC(BalancedRandomILSVRC, MultiWayClassification):
    name = 'balanced_rand_multiway_ilsvrc'


class FlatUnaryOriolMiniImageNet(FlatOriolMiniImageNet, UnaryClassification):
    name = 'flat_unary_oriol'


class FlatBinaryOriolMiniImageNet(FlatOriolMiniImageNet, BinaryClassification):
    name = 'flat_binary_oriol'


class FlatMultiWayOriolMiniImageNet(FlatOriolMiniImageNet, MultiWayClassification):
    name = 'flat_multiway_oriol'


class HierarchicalUnaryOriolMiniImageNet(HierarchicalOriolMiniImageNet, UnaryClassification):
    name = 'hier_unary_oriol'


class HierarchicalBinaryOriolMiniImageNet(HierarchicalOriolMiniImageNet, BinaryClassification):
    name = 'hier_binary_oriol'


class HierarchicalMultiWayOriolMiniImageNet(HierarchicalOriolMiniImageNet, MultiWayClassification):
    name = 'hier_multiway_oriol'


class RandomUnaryOriolMiniImageNet(RandomOriolMiniImageNet, UnaryClassification):
    name = 'rand_unary_oriol'


class RandomBinaryOriolMiniImageNet(RandomOriolMiniImageNet, BinaryClassification):
    name = 'rand_binary_oriol'


class RandomMultiWayOriolMiniImageNet(RandomOriolMiniImageNet, MultiWayClassification):
    name = 'rand_multiway_oriol'


class FlatUnarySachinMiniImageNet(FlatSachinMiniImageNet, UnaryClassification):
    name = 'flat_unary_sachin'


class FlatBinarySachinMiniImageNet(FlatSachinMiniImageNet, BinaryClassification):
    name = 'flat_binary_sachin'


class FlatMultiWaySachinMiniImageNet(FlatSachinMiniImageNet, MultiWayClassification):
    name = 'flat_multiway_sachin'


class HierarchicalUnarySachinMiniImageNet(HierarchicalSachinMiniImageNet, UnaryClassification):
    name = 'hier_unary_sachin'


class HierarchicalBinarySachinMiniImageNet(HierarchicalSachinMiniImageNet, BinaryClassification):
    name = 'hier_binary_sachin'


class HierarchicalMultiWaySachinMiniImageNet(HierarchicalSachinMiniImageNet, MultiWayClassification):
    name = 'hier_multiway_sachin'


class RandomUnarySachinMiniImageNet(RandomSachinMiniImageNet, UnaryClassification):
    name = 'rand_unary_sachin'


class RandomBinarySachinMiniImageNet(RandomSachinMiniImageNet, BinaryClassification):
    name = 'rand_binary_sachin'


class RandomMultiWaySachinMiniImageNet(RandomSachinMiniImageNet, MultiWayClassification):
    name = 'rand_multiway_sachin'

class BaselineILSVRC(FlatILSVRC, BaselineClassification):
    name = 'baseline_ilsvrc'


