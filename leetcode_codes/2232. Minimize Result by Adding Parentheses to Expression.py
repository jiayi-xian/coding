# 2232. Minimize Result by Adding Parentheses to Expression
# https://leetcode.com/problems/minimize-result-by-adding-parentheses-to-expression/description/
class Solution:
    def minimizeResult(self, expression: str) -> str:
        op_idx, n, res = expression.find('+'), len(expression), [float('inf'),expression] 
        def evaluate(exps: str):
            return eval(exps.replace('(','*(').replace(')', ')*').strip('*')) #  exps可能会出现‘*(12+34)*'所以要去掉所有 在外面的　‘*’
        for l in range(op_idx):
            for r in range(op_idx+1, n):
                exps = f'{expression[:l]}({expression[l:op_idx]}+{expression[op_idx+1:r+1]}){expression[r+1:n]}'
                val = evaluate(exps)
                if res[0] > val:
                    res[0], res[1] = val, exps
        return res[1]



    def maximizeResult(self, expression: str) -> str:
        op_idx, n, res = expression.find('+'), len(expression), [-float('inf'),expression] 
        def evaluate(exps: str):
            return eval(exps.replace('(','*(').replace(')', ')*').strip('*')) #  exps可能会出现‘*(12+34)*'所以要去掉所有 在外面的　‘*’
        for l in range(op_idx):
            for r in range(op_idx+1, n):
                exps = f'{expression[:l]}({expression[l:op_idx]}+{expression[op_idx+1:r+1]}){expression[r+1:n]}'
                val = evaluate(exps)
                if res[0] < val:
                    res[0], res[1] = val, exps
        return res[1]