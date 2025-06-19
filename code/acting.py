import torch
import libs.grammartools as gt

from libs.utils import GraphCountnodeDataset,Grammardataset
from torch_geometric.loader import DataLoader
from libs.grammar_model import rl_grammar
from tqdm import tqdm
import libs.MCTS as mcts
import libs.actor as act
import libs.training as tr
import gc
import tracemalloc
import sys
import argparse
import os



parser = argparse.ArgumentParser()


parser.add_argument('agent_nb',type=int)
parser.add_argument('ntask',type=int)
parser.add_argument('without_policy',default = 'False')
parser.add_argument('time_remain',type = int)

args = parser.parse_args()
nb_agent = args.agent_nb
without_policy = args.without_policy == 'True'
time_remain = args.time_remain

# tracemalloc.start()
ntask = args.ntask

def o(A,B,C,D,E):
  return torch.einsum("mik,mil,mlj,mkj,mkl->mij", A, B, C, D, E)

rules = """ 
        M ->  '('M'*'M')'  | 'o''('M','M','M','M','M')' |'('a'*'M '+' a'*'M')' | 'A' | 'I' | 'J'
        a ->  '1' | '-' '1'

        """ 
#a ->  '1''0' | '9' | '8' | '7' | '6' | '5' | '4' | '3' | '2' | '1' | '-''1''0' | '-''9' | '-''8' | '-''7' | '-''6' | '-''5' | '-''4' | '-''3' | '-''2' | '-''1'
#a ->  '1' | '-' '1'
#'('a'*'M '+' a'*'M')'

action_cost = {'o(M,M,M,M,M)':.00004,
                '(M*M)':.00002, }


G = gt.grammar(rules,action_cost)


path = 'save/agent'+str(nb_agent)
save_res = path + '/results_agent' + str(nb_agent) + '.dat'
save_data = path + '/save_agent' + str(nb_agent) + '.dat'

if not os.path.exists(path):
    os.makedirs(path)

graph_dataset = GraphCountnodeDataset(root="dataset/subgraphcount/")

split = int(len(graph_dataset))
rest_split = len(graph_dataset) - split

dt, rest_dt = torch.utils.data.random_split(graph_dataset,[split,rest_split])

graph_batch_size = 1024

lr_rl = 2*1e-6
lr_w = 1e-3

graph_loader = DataLoader(graph_dataset[dt.indices], batch_size=graph_batch_size, shuffle=True)


nb_episode = 1
nb_iter_mcts = 20000
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

max_depth = 100
nb_word = 1
batch_size = 512
value_confidence = 0.
search_param = 10000. #tester 100
bad_leaf = 10.

tree = mcts.Grammar_search(G,nb_word,nb_iter_mcts = nb_iter_mcts ,value_confidence = value_confidence ,search_param=search_param,bad_leaf=bad_leaf)
agent = act.actor(ntask)


device = torch.device('cpu')
rl_gram = rl_grammar(G,device,nb_word = nb_word,d_model = 128,nhead = 8,d_hid = 128, 
             num_layers = 5, dropout = 0.,max_depth = max_depth).to(device)

if not without_policy:
    rl_gram.load_state_dict(torch.load("save2/grammartest/grammar.dat",map_location=torch.device('cpu')))
    tree.value_confidence = .1

# print(tree.value_confidence)


rl_gram.eval()
with torch.no_grad():
    sequence = []
    max_reward = -100
    best_word = ['']
    
    tree.init_tree(rl_gram)
    root = tree.tree[tree.begin_word]
    seq,word,value = agent.episode(root,tree,rl_gram,graph_loader,without_policy,max_time =time_remain)
    sequence = agent.create_sequence(seq,tree)
    torch.save(sequence,save_data)
    if value>= max_reward:
        best_word = word
        max_reward = value
    message = "Agent {:04d}  score:{:6.6f} best score:{:6.4f} word:"
    with open(save_res,'w') as res:
        res.write(message.format(nb_agent,value,max_reward) + str(word) + "\n")








