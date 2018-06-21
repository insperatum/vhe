import math
import numpy as np
from scipy.stats import geom

import pregex as pre

maxDepth=2

class RegexPrior:
    def __init__(self):
        self.character_classes = [pre.d, pre.u, pre.l, pre.dot, pre.w, pre.s]
        self.p_regex = {
            pre.String: 0.2,
            **{k:0.3/6 for k in self.character_classes},
            **{k:0.5/5 for k in [pre.Concat, pre.Alt, pre.KleeneStar, pre.Plus, pre.Maybe]}
        }
        
        valid_no_recursion = [pre.String] + self.character_classes
        self.p_regex_no_recursion = \
            {k: self.p_regex[k] / sum(self.p_regex[k] for k in valid_no_recursion) if k in valid_no_recursion else 0 
            for k in self.p_regex}

        self.logp_regex = {k: math.log(p) if p>0 else float("-inf") for k,p in self.p_regex.items()}
        self.logp_regex_no_recursion = {k: math.log(p) if p>0 else float("-inf") for k,p in self.p_regex_no_recursion.items()}

    def sampleregex(self, depth=0):
        if depth < maxDepth:
            p_regex = self.p_regex
        else:
            p_regex = self.p_regex_no_recursion
        
        items = list(p_regex.items())
        idx = np.random.choice(range(len(items)), p=[p for k,p in items])
        R, p = items[idx]
            
        if R == pre.String:
            s = pre.Plus(pre.dot, p=0.3).sample()
            return R(s)
        elif R in self.character_classes:
            return R
        elif R in [pre.Concat, pre.Alt]:
            n = geom.rvs(0.8, loc=1)
            values = [self.sampleregex(depth+1) for i in range(n)]
            return R(values)
        elif R in [pre.KleeneStar, pre.Plus, pre.Maybe]:
            return R(self.sampleregex(depth+1))

    def scoreregex(self, r, depth=0):
        if depth<maxDepth:
            logp_regex = self.logp_regex
        else:
            logp_regex = self.logp_regex_no_recursion

        if r in self.character_classes:
            return logp_regex[r]
        else:
            R = type(r)
            p = logp_regex[R]
            if R == pre.String:
                return p + pre.Plus(pre.dot, p=0.3).match(r.arg)
            elif R == pre.Concat:
                n = len(r.values)
                return p + geom(0.8, loc=1).logpmf(n) + sum([self.scoreregex(s, depth=depth+1) for s in r.values])
            elif R == pre.Alt:
                n = len(r.values)
                if all(x==r.ps[0] for x in r.ps):
                    param_score = math.log(1/2)
                else:
                    param_score = math.log(1/2) - (len(r.ps)+1) #~AIC
                return p + geom(0.8, loc=1).logpmf(n) + param_score + sum([self.scoreregex(s, depth=depth+1) for s in r.values])
            elif R in [pre.KleeneStar, pre.Plus, pre.Maybe]:
                return p + self.scoreregex(r.val, depth=depth+1)
