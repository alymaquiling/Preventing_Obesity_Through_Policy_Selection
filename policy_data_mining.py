# ensure that all modules below are installed
# run with the following command:
# python policy_data_mining.py
# note that policy format is: health topic + type of group impacted

import pandas as pd
import itertools
import numpy as np
import pandas as pd
from collections import Counter,defaultdict
import numpy as np
from scipy.sparse import csr_matrix
import time

def loadData():
    file = 'legislation_data.csv'
    with open(file,'r',encoding='utf-8') as f:
        data = f.readlines()
        states = [a.split(',')[0] for a in data][1:]
        features = data[0][1:].split(',')[1:]
        state_data = [dict(zip(features,[int(num) for num in a.strip().split(',')[1:]])) for a in data[1:]]
        state_data = [[f for f in feature.keys() if feature[f] == 1] for feature in state_data]
        state_features = dict(zip(states,[s for s in state_data]))
        non_obesity = [['Obesity_Prevalence_Under_30_Percent'] + state_features[s] for s in state_features  if 'Obesity_Prevalence_Over_30_Percent' not in state_features[s] ]
        obesity = [ state_features[s] for s in state_features  if 'Obesity_Prevalence_Over_30_Percent' in state_features[s] ]
        state_data = obesity + non_obesity
        state_data_set = [s.strip() for ss in state_data for s in ss]
        cnt = dict(Counter(state_data_set))
        cnt_minsup = [c for c in cnt.keys() if cnt[c] >= 8]
        state_data = [[s for s in ss if s in cnt_minsup] for ss in state_data ]
        return state_data

# convert to true/false and return tuple of array and column for loading into pandas df
def fit_transform(X):
    unique_items = set()
    for transaction in X:
        for item in transaction:
            unique_items.add(item)
    columns_ = sorted(unique_items)
    columns_mapping = {}
    for col_idx, item in enumerate(columns_):
        columns_mapping[item] = col_idx
    columns_mapping_ = columns_mapping
    array = np.zeros((len(X), len(columns_)), dtype=bool)
    for row_idx, transaction in enumerate(X):
        for item in transaction:
            col_idx = columns_mapping_[item]
            array[row_idx, col_idx] = True
    return (array, columns_)

def setup_fptree(df, min_support):
    num_itemsets = len(df.index)
    is_sparse = False
    itemsets = df.values

    # support of each individual item
    # if itemsets is sparse, np.sum returns an np.matrix of shape (1, N)
    item_support = np.array(np.sum(itemsets, axis=0) / float(num_itemsets))
    item_support = item_support.reshape(-1)

    items = np.nonzero(item_support >= min_support)[0]

    # Define ordering on items for inserting into FPTree
    indices = item_support[items].argsort()
    rank = {item: i for i, item in enumerate(items[indices])}

    # Building tree by inserting itemsets in sorted order
    # Heuristic for reducing tree size is inserting in order
    #   of most frequent to least frequent
    tree = FPTree(rank)
    for i in range(num_itemsets):
        nonnull = np.where(itemsets[i, :])[0]
        itemset = [item for item in nonnull if item in rank]
        itemset.sort(key=rank.get, reverse=True)
        tree.insert_itemset(itemset)
    return tree, rank

def generate_itemsets(generator, num_itemsets, colname_map):
    itemsets = []
    supports = []
    for sup, iset in generator:
        itemsets.append(list(set(iset)))
        supports.append(sup / num_itemsets)

    res_df = pd.DataFrame({'support': supports, 'itemsets': itemsets})

    if colname_map is not None:
        res_df['itemsets'] = res_df['itemsets'] \
            .apply(lambda x: list(set([colname_map[i] for i in x])))
    return res_df

class FPTree(object):
    def __init__(self, rank=None):
        self.root = FPNode(None)
        self.nodes = defaultdict(list)
        self.cond_items = []
        self.rank = rank

    def conditional_tree(self, cond_item, minsup):
        # Find all path from root node to nodes for item
        branches = []
        count = defaultdict(int)
        for node in self.nodes[cond_item]:
            branch = node.itempath_from_root()
            branches.append(branch)
            for item in branch:
                count[item] += node.count

        # Define new ordering or deep trees may have combinatorially explosion
        items = [item for item in count if count[item] >= minsup]
        items.sort(key=count.get)
        rank = {item: i for i, item in enumerate(items)}

        # Create conditional tree
        cond_tree = FPTree(rank)
        for idx, branch in enumerate(branches):
            branch = sorted([i for i in branch if i in rank],
                            key=rank.get, reverse=True)
            cond_tree.insert_itemset(branch, self.nodes[cond_item][idx].count)
        cond_tree.cond_items = self.cond_items + [cond_item]

        return cond_tree

    def insert_itemset(self, itemset, count=1):
        self.root.count += count
        if len(itemset) == 0:
            return

        # Follow existing path in tree as long as possible
        index = 0
        node = self.root
        for item in itemset:
            if item in node.children:
                child = node.children[item]
                child.count += count
                node = child
                index += 1
            else:
                break

        # Insert any remaining items
        for item in itemset[index:]:
            child_node = FPNode(item, count, node)
            self.nodes[item].append(child_node)
            node = child_node

    def is_path(self):
        if len(self.root.children) > 1:
            return False
        for i in self.nodes:
            if len(self.nodes[i]) > 1 or len(self.nodes[i][0].children) > 1:
                return False
        return True

    def print_status(self, count, colnames):
        cond_items = [str(i) for i in self.cond_items]
        if colnames:
            cond_items = [str(colnames[i]) for i in self.cond_items]
        cond_items = ", ".join(cond_items)
        print('\r%d itemset(s) from tree conditioned on items (%s)' %
              (count, cond_items), end="\n")

class FPNode(object):
    def __init__(self, item, count=0, parent=None):
        self.item = item
        self.count = count
        self.parent = parent
        self.children = defaultdict(FPNode)

        if parent is not None:
            parent.children[item] = self

    def itempath_from_root(self):
        path = []
        if self.item is None:
            return path

        node = self.parent
        while node.item is not None:
            path.append(node.item)
            node = node.parent

        path.reverse()
        return path

def fpgrowth(df, min_support=0.5, use_colnames=False, max_len=None):
    colname_map = None
    if use_colnames:
        colname_map = {idx: item for idx, item in enumerate(df.columns)}

    tree, _ = setup_fptree(df, min_support)
    minsup = int(min_support * len(df.index) + 1)  # min support as count
    generator = fpg_step(tree, minsup, colname_map, max_len)

    return generate_itemsets(generator, len(df.index), colname_map)

def fpg_step(tree, minsup, colnames, max_len):
    count = 0
    items = tree.nodes.keys()
    if tree.is_path():
        # If the tree is a path, we can combinatorally generate all
        # remaining itemsets without generating additional conditional trees
        size_remain = len(items) + 1
        if max_len:
            size_remain = max_len - len(tree.cond_items) + 1
        for i in range(1, size_remain):
            for itemset in itertools.combinations(items, i):
                count += 1
                support = min([tree.nodes[i][0].count for i in itemset])
                yield support, tree.cond_items + list(itemset)
    elif not max_len or max_len > len(tree.cond_items):
        for item in items:
            count += 1
            support = sum([node.count for node in tree.nodes[item]])
            yield support, tree.cond_items + [item]

    # Generate conditional trees to generate frequent itemsets one item larger
    if not tree.is_path() and (not max_len or max_len > len(tree.cond_items)):
        for item in items:
            cond_tree = tree.conditional_tree(item, minsup)
            for sup, iset in fpg_step(cond_tree, minsup,colnames, max_len):
                yield sup, iset

print('Problem statement: Obesity rates are increasing and this is an indicator that health issues are worsening in American adults.\n')
time.sleep(4)
print('About half of the 50 states had an obesity rate of 30% in American adults in 2018.\n')
time.sleep(3)
print('Our goal is to observe the policies enacted regarding health in each of the 50 states and determine ')
print('which sets of policies can be applied to the states with higher obesity rates.\n')
time.sleep(5)
print('To achieve this, we will use FP-growth to find itemsets of policies with k = 3 since each state has an limited budget')
print('and we would like to find the policies that have the greatest potential impact so that we are provided with a list of')
print('policies that we may later suggest to local lawmakers.\n-------------------------------------------------------------------------')
time.sleep(7)
print('loading data for all states\n')
time.sleep(2)
stateObesity = loadData()

print('separating states with obesity rates over and under 30%\n')
obese = [s for s in stateObesity if 'Obesity_Prevalence_Under_30_Percent' in s]
non_obese = [s for s in stateObesity if 'Obesity_Prevalence_Over_30_Percent' in s]

print('fitting and transforming data\n')
tup = fit_transform(obese)
dat = [t[1:] for t in tup[0]]
col = tup[1][1:]
df = pd.DataFrame(dat, columns=col)
print('applying fp-growth to states with higher obesity rates\n')
obese_data = fpgrowth(df, min_support=0.6, use_colnames=True,max_len=3)
print('removing itemsets if k != 3\n')
obese_data['NUM_ITEMS'] = obese_data['itemsets'].apply(len)
obese_data = obese_data.loc[obese_data['NUM_ITEMS'] == 3]
obese_data = obese_data[['support','itemsets']]

print('fitting and transforming data\n')
tup = fit_transform(non_obese)
dat = [t[1:] for t in tup[0]]
col = tup[1][1:]
df = pd.DataFrame(dat, columns=col)
print('applying fp-growth to states with lower obesity rates\n')
non_obese_data = fpgrowth(df, min_support=0.6, use_colnames=True,max_len=3)
print('removing itemsets if k != 3\n')
non_obese_data['NUM_ITEMS'] = non_obese_data['itemsets'].apply(len)
non_obese_data = non_obese_data.loc[non_obese_data['NUM_ITEMS'] == 3]
non_obese_data = non_obese_data[['support','itemsets']]

print('removing itemsets that are common between two sets')
print('please be patient...this will take a minute!')
non_obese_data['not_in_obese_set'] = non_obese_data['itemsets']\
                                                        .apply(lambda item: set(item) not in [set(o) for o in obese_data.itemsets])

print('formatting data for output\n')
non_obese_data['Policy 1'] = [i[0].replace('Policy_','').replace('_',' ') for i in non_obese_data.itemsets]
non_obese_data['Policy 2'] = [i[1].replace('Policy_','').replace('_',' ') for i in non_obese_data.itemsets]
non_obese_data['Policy 3'] = [i[2].replace('Policy_','').replace('_',' ') for i in non_obese_data.itemsets]
pd.set_option('display.width', 180)
pd.set_option('display.max_columns', 30)
pd.set_option('display.max_rows', 1000)

final_set = non_obese_data.loc[non_obese_data['not_in_obese_set']][['support','Policy 1','Policy 2','Policy 3']]
print(final_set)
