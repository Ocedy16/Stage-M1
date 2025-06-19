import torch
import numpy as np
import psutil as ps
from itertools import combinations, permutations
import time


class Node(object):
    def __init__(self,parent,var,position,state,policy,value,leaf,cost,root = False,bad_leaf = False):
        super().__init__()
        self.root = root
        self.leaf = leaf
        self.parent = [parent] if parent is not None else parent
        self.position = position
        self.state = state
        self.children = {}
        self.n = torch.tensor(0.)
        self.R = torch.tensor(0.)
        self.value = value
        self.policy = policy
        self.var = var
        self.cost = cost
        self.bad_leaf = bad_leaf

class Grammar_search(object):
    def __init__(self,grammar,nb_word,max_length = 30,nb_iter_mcts = 1000,value_confidence = 0.,search_param = 2.0**.5,bad_leaf = 100):
        super().__init__()
        self.grammar = grammar
        
        self.end = [grammar.dict_wtv[v] for v in grammar.variable]
    
        self.begin =  [torch.zeros((1,1),dtype = torch.int32)]*nb_word
        self.begin_word = str(sorted(self.grammar.vec_to_word(self.begin)))
        
        index = index = -torch.ones(len(grammar.dict_vtw)) * float('inf')

        self.nb_iter_mcts = nb_iter_mcts
        self.value_confidence = value_confidence
        self.nb_word = nb_word
        self.max_length = max_length
        self.explore = {}
        self.explore_terminal = {}
        self.search_param = search_param
        self.bad_leaf = bad_leaf
        
        for v in grammar.variable:
            tmp = index.clone()
            tmp2 = index.clone()
            tmp[torch.tensor(grammar.index_variable_rule[grammar.dict_wtv[v]])] = 0
            tmp2[torch.tensor(grammar.index_variable_terminal_rule[grammar.dict_wtv[v]])] = 0
            self.explore[grammar.dict_wtv[v]] = torch.softmax(tmp,0)
            self.explore_terminal[grammar.dict_wtv[v]] = torch.softmax(tmp2,0)
        self.nb_iter_mcts = nb_iter_mcts
        self.tree = {}
        
    def init_tree(self,rl_gram):
        policy,value = rl_gram(self.begin[0],torch.cat(self.begin),1)
        position,leaf = self.position_in_word(self.begin)
        self.tree.clear()
        self.tree = {self.begin_word:Node(None,self.begin[0],position,self.begin,policy,value,False,0,root = True)}
    
    def is_word(self,w):
        ret = torch.where(self.cond(w))
        return ret[0].shape[0]<1
    
    def cond(self,x):
        ret = False
        for val in self.end:
            ret += x == val
        return ret
    
    def position_in_word(self,state):
        for i,st in enumerate(state):
            ret = torch.where(self.cond(st.T))
            if ret[0].shape[0] > 0:
                ret = (ret[1][0:1],ret[0][0:1]+i)

                return ret,True
            
        ret = (torch.tensor([0]),torch.tensor([0]))
        return ret,False
    

        
    
    
    def MCTS(self,root,rl_gram,loader,ntask,nb_test = 10000,memory_min_free = .25,without_policy = False,max_time = 24*3600 ):
        debut = time.time()
        for i in range(nb_test):
            if  time.time() - debut >=max_time:
                break
            leaf = self.select(root,rl_gram,without_policy = without_policy)
            sim_res = self.rollout(leaf,loader,ntask)
            self.backprop(leaf,sim_res)
        prob = root.policy*0
        tau = self.iprob(root.n)
        sum_prob = 0
        for child in root.children.values():
            sum_prob += self.tree[child].n**tau
        for act in root.children.keys():
            prob[act] = self.tree[root.children[act]].n**tau/(sum_prob)
        return self.best_child(root,without_policy=without_policy,selection=True),prob
    

    def select(self,node,rl_gram,without_policy):
        while self.is_expand(node):
            node = self.tree[node.children[self.best_child(node,without_policy=without_policy)]]

        return self.expand(node,rl_gram,without_policy=without_policy)
    
    def is_expand(self,node):
        if node.position[1].shape[0]<1:
            return False
        if node.state[node.position[1]].shape[0]< self.max_length:
            return len(node.children) == len(self.grammar.index_variable_rule[node.var.item()])
        return len(node.children) == len(self.grammar.index_variable_terminal_rule[node.var.item()])
   
    def expand(self,node,rl_gram,without_policy):
        if node.leaf:
            return node
        explore = self.explore[node.var.item()].clone()
        if node.state[node.position[1]].shape[0]> self.max_length:
            explore = self.explore_terminal[node.var.item()].clone()
        for act in node.children.keys():
            explore[act] = 0
        #print(node.children.keys())
        #print(explore)
        total = explore.sum()
        explore = explore / total
        action = torch.multinomial(explore,1)
        action_cost = node.cost
        if action.item() in self.grammar.action_cost:
            action_cost += self.grammar.action_cost[action.item()]
        state = [st.clone() for st in node.state]
        if action == self.grammar.epsilon:
            del state[node.position[1]]
            if len(state) == 0:
                state = [torch.tensor([[self.grammar.epsilon]])]
                bad_leaf = True
            else:
                bad_leaf = False
        else:
            rule = self.grammar.word_to_vec(self.grammar.dict_vtw[action.item()],len(self.grammar.dict_vtw[action.item()]))
            state[node.position[1]] = torch.cat([state[node.position[1]][:node.position[0],:],rule,state[node.position[1]][node.position[0]+1:,:]])
            bad_leaf = False
        parent = str(sorted(self.grammar.vec_to_word(node.state)))
        expand_key = str(sorted(self.grammar.vec_to_word(state)))
        if expand_key in self.tree:
            self.tree[expand_key].parent.append(parent)
            node.children[action.item()] = expand_key
            return self.select(node,rl_gram,without_policy = without_policy)
        else:
            position,leaf = self.position_in_word(state)
            if leaf:
                var = state[position[1].item()][position[0]]
                policy,value = rl_gram(var,torch.cat(state),state[position[1].item()].shape[0])
            else:
                var = node.var
                policy ,value = rl_gram(var,torch.cat(state),state[position[1].item()].shape[0])
            
            if not bad_leaf and position[1].item()>0 and self.is_word(state[position[1].item()]):
                for st in state[:position[1].item()]:
                    if st.shape[0] ==  state[position[1].item()].shape[0] and torch.abs(st-state[position[1].item()]).sum()==0:
                        leaf = False
                        bad_leaf = True
                        break
                
            
            
            
            new = Node(parent,var,position,state,policy,value.detach(),not leaf,action_cost,bad_leaf = bad_leaf)
            if bad_leaf:
                new.R = torch.tensor(-self.bad_leaf)
            self.tree[expand_key] = new
            node.children[action.item()] = expand_key
            return new
    
    def rollout(self,leaf,loader,ntask):
        if leaf.bad_leaf:
            return -self.bad_leaf
        tr = not leaf.leaf
        if tr:
            var = leaf.var.clone()
        state = [st.clone() for st in leaf.state]
        leaf_key = str(sorted(self.grammar.vec_to_word(state)))
        if leaf.leaf and self.tree[leaf_key].n >0:
            return self.tree[leaf_key].R/self.tree[leaf_key].n
        position = leaf.position
        action_cost = leaf.cost
        while tr:
            explore = self.explore[var.item()]
            if state[position[1].item()].shape[0]> self.max_length:
                explore = self.explore_terminal[var.item()]
            action = torch.multinomial(explore,1)
            if action == self.grammar.epsilon:
                del state[position[1].item()]
                if len(state) == 0:
                    return -self.bad_leaf
            else:
                act_id = action.item()
                if act_id in self.grammar.action_cost:
                    action_cost += self.grammar.action_cost[act_id]
                rule = self.grammar.word_to_vec(self.grammar.dict_vtw[act_id],len(self.grammar.dict_vtw[act_id]))
                state[position[1]] = torch.cat([state[position[1]][:position[0],:],rule,state[position[1]][position[0]+1:,:]])
                if position[1].item()>0 and self.is_word(state[position[1].item()]):
                    for st in state[:position[1].item()]:
                        if st.shape[0] ==  state[position[1].item()].shape[0] and torch.abs(st-state[position[1].item()]).sum()==0:
                            return -self.bad_leaf - action_cost
            position,tr = self.position_in_word(state)
            if tr:
                var = state[position[1]][position[0]]
        return self.result(state,loader,ntask) - action_cost #- .001*len(state)
    
    def result(self,state,loader,ntask):
        word = self.grammar.vec_to_word(state)
        #word = ['o(A,J-A,A,J-A,A)*(J-A)']
        L = 0
        nb = 0
        for data in loader:
            data= data
            out = torch.stack([
                        self.grammar.calculatrice(w, data.A, data.I, data.J)
                                for w in word
                                    ], dim=1)
            target = data.y[:, ntask, :, :]  # shape: [batch, n_nodes, n_features]
            diff = torch.abs(out - target.unsqueeze(1))  # shape: [batch, n_rules, n_nodes, n_features]
            #diff2 = (torch.abs(out[torch.where(out != 0)] - target.unsqueeze(1)[torch.where(out != 0)])**4).mean()
            norm = (data.n_node.unsqueeze(1).unsqueeze(2)) ** 2
            score = (diff / norm).sum((2, 3)) 
            L += score.mean()
            nb += 1
            if torch.all(out*(data.J) == 0) or torch.all(out-data.A.unsqueeze(1)==0) or torch.all(out-data.J.unsqueeze(1)==0) or torch.all(out-(data.J.unsqueeze(1)-data.A.unsqueeze(1))==0): 
                #print("it's zero")
                return -self.bad_leaf/10000 #chercher à moins pénaliser peut être

        if nb > 0:
            t_ret = 100 * torch.exp(-L / nb *6)  # 4 est un hyperparamètre de température
        else:
            t_ret = -self.bad_leaf

        print(word, t_ret)
        return t_ret
    
    def backprop(self,node,result,gamma = 1.):
        if node.root:
            return
        node.n += 1
        node.R += result
        
        for key in node.parent:
            self.backprop(self.tree[key],gamma*(result))
    
    def iprob(self,n,N = 15000000 ):
        if n>N:
            return np.log(N)/np.log(n)
        else:
            return 1.
    
    def best_child(self,node,without_policy,selection =False):
        sum_n =  0 
        sum_prob = 0
        tree_conf = 1-self.value_confidence
        tau = self.iprob(node.n)
        for child in node.children.values():
            sum_n += self.tree[child].n
            
        if selection:
            choices_weights = [tree_conf*self.tree[node.children[k]].R/max(self.tree[node.children[k]].n,1)
                               +self.value_confidence*self.tree[node.children[k]].value
                               # + self.search_param*np.sqrt(sum_n)/(self.tree[node.children[k]].n+1)
                               for k in list(node.children.keys())]
            ret = torch.argmax(torch.tensor(choices_weights))
            action = list(node.children.keys())[ret]
            return action

        if without_policy:
            choices_weights = [tree_conf*self.tree[node.children[k]].R/max(self.tree[node.children[k]].n,1)
                               +self.value_confidence*self.tree[node.children[k]].value
                               + self.search_param*np.sqrt(sum_n)/(self.tree[node.children[k]].n+1)
                               for k in list(node.children.keys())]
        else:
            choices_weights = [tree_conf*self.tree[node.children[k]].R/max(self.tree[node.children[k]].n,1)
                           +self.value_confidence*self.tree[node.children[k]].value
                           + self.search_param*node.policy[k]*np.sqrt(sum_n)/(self.tree[node.children[k]].n+1)
                           for k in list(node.children.keys())]
        ret = torch.argmax(torch.tensor(choices_weights))
        action = list(node.children.keys())[ret]
        return action
            
    def suppr_sub_tree(self,node):
        if len(self.tree[node].parent)>0:
            return
        else:
            if len(self.tree[node].children)>0:
                l_children = list(self.tree[node].children.values())
                for child in l_children:

                    del(self.tree[child].parent[self.tree[child].parent.index(node)])
                    self.suppr_sub_tree(child)
            del self.tree[node]
            
        
            
        
            
        


    

