import jiwer

class Wer():
    def __init__(self, apply_normalizate:bool = False, list_normalizate: list = [],return_comparate:bool = False, debug:bool = False) -> None:
        
        self.apply_normalizate = apply_normalizate

        if apply_normalizate:
            assert list_normalizate == [], "Need list of transformatio to contiune"
            self.transform = jiwer.Compose([list_normalizate])

        self.return_comparate = return_comparate
        self.debug = debug

    def evaluate(self, ref, hyp):

        if self.apply_normalizate:
            ref = self.transform(ref)
            hyp = self.transform(hyp)

        r = ref.split()
        h = hyp.split()
        #costs will holds the costs, like in the Levenshtein distance algorithm
        costs = [[0 for inner in range(len(h)+1)] for outer in range(len(r)+1)]
        # backtrace will hold the operations we've done.
        # so we could later backtrace, like the WER algorithm requires us to.
        backtrace = [[0 for inner in range(len(h)+1)] for outer in range(len(r)+1)]

        OP_OK = 0
        OP_SUB = 1
        OP_INS = 2
        OP_DEL = 3

        DEL_PENALTY=1 # Tact
        INS_PENALTY=1 # Tact
        SUB_PENALTY=1 # Tact
        # First column represents the case where we achieve zero
        # hypothesis words by deleting all reference words.
        for i in range(1, len(r)+1):
            costs[i][0] = DEL_PENALTY*i
            backtrace[i][0] = OP_DEL

        # First row represents the case where we achieve the hypothesis
        # by inserting all hypothesis words into a zero-length reference.
        for j in range(1, len(h) + 1):
            costs[0][j] = INS_PENALTY * j
            backtrace[0][j] = OP_INS

        # computation
        for i in range(1, len(r)+1):
            for j in range(1, len(h)+1):
                if r[i-1] == h[j-1]:
                    costs[i][j] = costs[i-1][j-1]
                    backtrace[i][j] = OP_OK
                else:
                    substitutionCost = costs[i-1][j-1] + SUB_PENALTY # penalty is always 1
                    insertionCost    = costs[i][j-1] + INS_PENALTY   # penalty is always 1
                    deletionCost     = costs[i-1][j] + DEL_PENALTY   # penalty is always 1

                    costs[i][j] = min(substitutionCost, insertionCost, deletionCost)
                    if costs[i][j] == substitutionCost:
                        backtrace[i][j] = OP_SUB
                    elif costs[i][j] == insertionCost:
                        backtrace[i][j] = OP_INS
                    else:
                        backtrace[i][j] = OP_DEL

        # back trace though the best route:
        i = len(r)
        j = len(h)
        numSub = 0
        numDel = 0
        numIns = 0
        numCor = 0
        if self.debug:
            lines = []
            compares = []
        while i > 0 or j > 0:
            if backtrace[i][j] == OP_OK:
                numCor += 1
                i-=1
                j-=1
                if self.debug:
                    lines.append("OK\t" + r[i]+"\t"+h[j])
                    compares.append(colored(0, 0, 0, h[j]))
            elif backtrace[i][j] == OP_SUB:
                numSub +=1
                i-=1
                j-=1
                if self.debug:
                    lines.append("SUB\t" + r[i]+"\t"+h[j])
                    compares.append(colored(0, 255, 0, h[j]) +  colored(0, 0, 0, f'({r[i]})'))
            elif backtrace[i][j] == OP_INS:
                numIns += 1
                j-=1
                if self.debug:
                    lines.append("INS\t" + "****" + "\t" + h[j])
                    compares.append(colored(0, 0, 255, h[j]))
            elif backtrace[i][j] == OP_DEL:
                numDel += 1
                i-=1
                if self.debug:
                    lines.append("DEL\t" + r[i]+"\t"+"****")
                    compares.append(colored(255, 0, 0, r[i]))
        if self.debug:
            compares = compares[::-1]
            for line in compares:
                print(line, end=" ")

        wer_result = round( (numSub + numDel + numIns) / (float) (len(r)), 3)
        if self.return_comparate:
            return {'WER':wer_result, 'Cor':numCor, 'Sub':numSub, 'Ins':numIns, 'Del':numDel}, compares
        else:
            return {'WER':wer_result, 'Cor':numCor, 'Sub':numSub, 'Ins':numIns, 'Del':numDel}


def colored(r, g, b, text):
    return "\033[38;2;{};{};{}m{} \033[38;2;255;255;255m".format(r, g, b, text)

def strike(text, color=None):
    if color:
      return colored(0, 255, 0, ''.join([u'\u0336{}'.format(c) for c in text]))
      
    else:
      return  colored(0, 0, 0, ''.join([u'\u0332{}'.format(c) for c in text]))


