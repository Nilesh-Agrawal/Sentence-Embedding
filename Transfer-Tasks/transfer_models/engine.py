'''

Generic sentence evaluation scripts wrapper

'''
from __future__ import absolute_import, division, unicode_literals

from senteval import utils
from senteval.binary import CREval, MREval, MPQAEval, SUBJEval
from senteval.snli import SNLIEval
from senteval.sick import SICKRelatednessEval, SICKEntailmentEval
from senteval.sts import STS12Eval, STS13Eval, STS14Eval, STS15Eval, STS16Eval, STSBenchmarkEval
from senteval.sst import SSTEval


class SE(object):
    def __init__(self, params, batcher, prepare=None):
        # parameters
        params = utils.dotdict(params)
        params.usepytorch = True if 'usepytorch' not in params else params.usepytorch
        params.seed = 1111 if 'seed' not in params else params.seed

        params.batch_size = 128 if 'batch_size' not in params else params.batch_size
        params.nhid = 0 if 'nhid' not in params else params.nhid
        params.kfold = 5 if 'kfold' not in params else params.kfold

        if 'classifier' not in params or not params['classifier']:
            params.classifier = {'nhid': 0}

        assert 'nhid' in params.classifier, 'Set number of hidden units in classifier config!!'

        self.params = params

        # batcher and prepare
        self.batcher = batcher
        self.prepare = prepare if prepare else lambda x, y: None

        self.list_tasks = ['CR', 'MR', 'MPQA', 'SUBJ', 'SST2', 'SST5',
                           'SICKRelatedness', 'SICKEntailment', 'STSBenchmark',
                           'SNLI', 'STS12', 'STS13',
                           'STS14', 'STS15', 'STS16']

    def eval(self, name):
        # evaluate on evaluation [name], either takes string or list of strings
        if (isinstance(name, list)):
            self.results = {x: self.eval(x) for x in name}
            return self.results

        tpath = self.params.task_path
        assert name in self.list_tasks, str(name) + ' not in ' + str(self.list_tasks)
        if name == 'CR':
            self.evaluation = CREval(tpath + '/CR', seed=self.params.seed)
        elif name == 'MR':
            self.evaluation = MREval(tpath + '/MR', seed=self.params.seed)
        elif name == 'MPQA':
            self.evaluation = MPQAEval(tpath + '/MPQA', seed=self.params.seed)
        elif name == 'SUBJ':
            self.evaluation = SUBJEval(tpath + '/SUBJ', seed=self.params.seed)
        elif name == 'SST2':
            self.evaluation = SSTEval(tpath + '/SST/binary', nclasses=2, seed=self.params.seed)
        elif name == 'SST5':
            self.evaluation = SSTEval(tpath + '/SST/fine', nclasses=5, seed=self.params.seed)
	elif name == 'SICKRelatedness':
            self.evaluation = SICKRelatednessEval(tpath + '/SICK', seed=self.params.seed)
        elif name == 'STSBenchmark':
            self.evaluation = STSBenchmarkEval(tpath + '/STS/STSBenchmark', seed=self.params.seed)
        elif name == 'SICKEntailment':
            self.evaluation = SICKEntailmentEval(tpath + '/SICK', seed=self.params.seed)
        elif name == 'SNLI':
            self.evaluation = SNLIEval(tpath + '/SNLI', seed=self.params.seed)
        elif name in ['STS12', 'STS13', 'STS14', 'STS15', 'STS16']:
            fpath = name + '-en-test'
            self.evaluation = eval(name + 'Eval')(tpath + '/STS/' + fpath, seed=self.params.seed)
        
        self.params.current_task = name
        self.evaluation.do_prepare(self.params, self.prepare)

        self.results = self.evaluation.run(self.params, self.batcher)

        return self.results
